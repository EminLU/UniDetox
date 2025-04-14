#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
toxic_gpt2_finetune_and_distill.py

This script demonstrates how to:
  1) Filter and fine-tune a GPT-2 model to obtain a "toxic" model
  2) Generate detoxifying text by contrastive decoding using a base GPT-2 model
     and the newly fine-tuned "toxic" model.
"""

import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softmax
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          Trainer, TrainingArguments,
                          DataCollatorWithPadding, 
                          GPT2LMHeadModel, GPT2Tokenizer, GPT2Model, GPT2Config, 
                          pipeline, set_seed, Trainer, TrainingArguments, )
from transformers.modeling_outputs import (CausalLMOutputWithCrossAttentions, CausalLMOutputWithPast)
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from typing import Optional, Tuple, Union, List
from tqdm import tqdm






###################################################################
# 1) Filtering and Fine-Tuning on a Subset of DGHS (to get "toxic" model)
###################################################################

def load_and_filter_dghs(
    auth_token: str, 
    targets_of_interest=None
):
    """
    Load DGHS dataset and filter for hateful text in specified target categories.

    :param auth_token: HuggingFace token for 'LennardZuendorf/Dynamically-Generated-Hate-Speech-Dataset'
    :param targets_of_interest: list of strings specifying which groups to keep
    :return: train/val/test splits of the filtered dataset
    """
    if targets_of_interest is None:
        targets_of_interest = [
            'wom','trans','gendermin','bis','gay','gay.man','gay.wom',
            'mixed.race','ethnic.minority','indig','indig.wom','non.white','bla','bla.wom','bla.man',
            'asi','asi.wom','asi.east','asi.south','asi.chin','asi.pak','arab',
            'eastern.europe','russian','pol','hispanic','immig','asylum','ref','for',
            'jew','mus','mus.wom','other.religion'
        ]

    # Load
    dghs = load_dataset("LennardZuendorf/Dynamically-Generated-Hate-Speech-Dataset",
                         token=auth_token)

    # Convert to filter
    def filter_func(example):
        return example['target'] in targets_of_interest and example['label'] == 'hate'

    dghs_filtered = dghs['train'].filter(filter_func)
    
    # The dataset also uses a 'split' column for train/val/test
    train_data = dghs_filtered.filter(lambda ex: ex['split']=='train')
    val_data = dghs_filtered.filter(lambda ex: ex['split']=='dev')
    test_data = dghs_filtered.filter(lambda ex: ex['split']=='test')

    print(f"train size: {len(train_data)}")
    print(f"val size:   {len(val_data)}")
    print(f"test size:  {len(test_data)}")

    return train_data, val_data, test_data

def create_encoded_datasets(
    train_data, val_data,
    tokenizer, 
    max_length=512
):
    """
    Encode the text fields of train/val for causal LM fine-tuning.
    """
    toxicity_train = np.array([ex['text'] for ex in train_data])
    toxicity_val   = np.array([ex['text'] for ex in val_data])

    train_encodings = tokenizer(
        toxicity_train.tolist(),
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=max_length
    )
    val_encodings = tokenizer(
        toxicity_val.tolist(),
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=max_length
    )

    # Convert to list-of-dicts for Trainer
    train_dataset = []
    for inp, mask in zip(train_encodings['input_ids'], train_encodings['attention_mask']):
        train_dataset.append({
            'input_ids': inp,
            'attention_mask': mask,
            'labels': inp
        })

    val_dataset = []
    for inp, mask in zip(val_encodings['input_ids'], val_encodings['attention_mask']):
        val_dataset.append({
            'input_ids': inp,
            'attention_mask': mask,
            'labels': inp
        })

    return train_dataset, val_dataset

def fine_tune_toxic_model(
    base_model_name: str,
    auth_token: str,
    output_dir: str,
    epochs=3,
    lr=1e-5,
    batch_size=16,
    random_seed=42
):
    """
    1) Loads DGHS and filters it
    2) Creates train/val encodings
    3) Fine-tunes a base model on that toxic dataset
    4) Saves to output_dir

    Example usage:
        fine_tune_toxic_model(
            base_model_name='gpt2-xl',
            auth_token='hf_...',
            output_dir='./gpt2_toxic_model/'
        )
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # 1. Load and filter
    train_data, val_data, _ = load_and_filter_dghs(auth_token)

    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Encode
    train_dataset, val_dataset = create_encoded_datasets(train_data, val_data, tokenizer)

    # 4. Model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        use_auth_token=None,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    # 5. Trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        save_total_limit=1,
        seed=random_seed
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )

    trainer.train()
    print("[fine_tune_toxic_model] Training complete!")
    # Trainer will save final model to output_dir

###################################################################
# 2) Generating Detoxifying Text by Contrastive Decoding
###################################################################

class DetoxifiedGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config, toxic_model_path, alpha=0.1, beta=1.0):
        super().__init__(config)
        # Load the toxic model separately
        self.toxic_model = GPT2LMHeadModel.from_pretrained(toxic_model_path)
        self.alpha = alpha
        self.beta = beta
        
    def verify_model_parameters(self, toxic_model_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(toxic_model_path + '/pytorch_model.bin', map_location=device)
        self.toxic_model.to(device)
        toxic_model_state_dict = self.toxic_model.state_dict()
        parameters_match = True
        
        for param_tensor in checkpoint:
            if param_tensor in toxic_model_state_dict:
                if checkpoint[param_tensor].shape != toxic_model_state_dict[param_tensor].shape:
                    print(f"Shape mismatch at {param_tensor}: checkpoint={checkpoint[param_tensor].shape}, loaded={toxic_model_state_dict[param_tensor].shape}")
                    parameters_match = False
                elif not torch.equal(checkpoint[param_tensor], toxic_model_state_dict[param_tensor]):
                    print(f"Value mismatch at {param_tensor}")
                    parameters_match = False
            else:
                print(f"Parameter {param_tensor} found in checkpoint but not in loaded model")
                parameters_match = False

        for param_tensor in toxic_model_state_dict:
            if param_tensor not in checkpoint:
                print(f"Parameter {param_tensor} found in loaded model but not in checkpoint")
                parameters_match = False
        
        return parameters_match
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        # Forward pass on the original model
        original_outputs = super().forward(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = original_outputs[0]
        
        # Forward pass on the toxic model
        with torch.no_grad():
            toxic_outputs = self.toxic_model(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        
        base_logits = original_outputs.logits
        toxic_logits = toxic_outputs.logits

        # Calculate the cutoff
        cutoff = torch.log(torch.tensor(self.alpha)) + base_logits.max(dim=-1, keepdim=True).values
        
        # Calculate the diffs
        # diffs = (1 + self.beta) * base_logits - self.beta * toxic_logits
        diffs = base_logits - toxic_logits
        
        # Mask logits
        cd_logits = diffs.masked_fill(base_logits < cutoff, -float('inf'))
        
        # Continue with the loss calculation or return the cd logits
        loss = None
        if labels is not None:
            labels = labels.to(cd_logits.device)
            shift_logits = cd_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (cd_logits,) + original_outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=cd_logits,
            past_key_values=original_outputs.past_key_values,
            hidden_states=original_outputs.hidden_states,
            attentions=original_outputs.attentions,
            cross_attentions=original_outputs.cross_attentions,
        )

def generate_detox_text(
    base_model_name: str,
    toxic_model_ckpt: str,
    output_csv_dir: str,
    alpha_list=None,
    beta_list=None,
    random_seed=42
):
    """
    Generate text from a 'DetoxifiedGPT2LMHeadModel' that does approximate 
    contrastive decoding, storing text in CSV files.

    :param base_model_name: the base GPT-2 model name, e.g. 'gpt2-xl'
    :param toxic_model_ckpt: path to the fine-tuned toxic GPT-2 model
    :param output_csv_dir: where to store generated CSVs
    :param alpha_list: list of alpha hyperparams
    :param beta_list: list of beta hyperparams
    :param random_seed: for reproducibility
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    if alpha_list is None:
        alpha_list = [0.1]   # example
    if beta_list is None:
        beta_list = [1.0]    # example

    os.makedirs(output_csv_dir, exist_ok=True)

    # load base GPT2
    tokenizer = GPT2Tokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print("[generate_detox_text] loaded base model tokenizer:", base_model_name)

    # For generation, we can start with BOS token
    if tokenizer.bos_token is None:
        bos_id = tokenizer.eos_token_id
    else:
        bos_id = tokenizer.bos_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for alpha in alpha_list:
        for beta in beta_list:
            print(f"[generate_detox_text] alpha={alpha}, beta={beta}")
            # load a special "DetoxifiedGPT2LMHeadModel"
            model = DetoxifiedGPT2LMHeadModel.from_pretrained(
                base_model_name,
                toxic_model_path=toxic_model_ckpt,
                alpha=alpha,
                beta=beta, 
                torch_dtype=torch.bfloat16,
                # device_map='auto'
            )
            
            model.to(device)

            # create a small prompt
            encoded_inputs = tokenizer(tokenizer.bos_token, return_tensors='pt')
            input_ids = encoded_inputs['input_ids'].to(device)
            attention_mask = encoded_inputs['attention_mask'].to(device)

            all_texts = []
            # e.g. generate 5 times * 128 sequences each
            for _ in range(5):
                out_seqs = model.generate(
                    input_ids=input_ids,
                    pad_token_id=tokenizer.eos_token_id, 
                    attention_mask=attention_mask, 
                    max_length=256,
                    num_return_sequences=128,
                    do_sample=True
                )
                for seq in out_seqs:
                    text = tokenizer.decode(seq, skip_special_tokens=True)
                    all_texts.append(text)

            # store CSV
            ckpt_name = os.path.basename(toxic_model_ckpt)
            csv_name = f"generated_texts_alpha={alpha}_beta={beta}_{ckpt_name}.csv"
            csv_path = os.path.join(output_csv_dir, csv_name)
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Text"])
                for txt in all_texts:
                    writer.writerow([txt])
            print(f"Saved detoxifying text => {csv_path}")

import argparse

def main():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on toxic data and generate detox text.")
    parser.add_argument("--base_model_name", type=str, default="gpt2-xl", help="Which model to use.")
    parser.add_argument("--output_dir", type=str, default="./toxic_model_output", help="Where to save the toxic model.")
    parser.add_argument("--auth_token", type=str, default=None, help="Hugging Face auth token if needed.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--random_seed_training", type=int, default=123)
    parser.add_argument("--random_seed_distilling", type=int, default=1234)
    parser.add_argument("--alpha_list", nargs="+", default=["0.1"])
    parser.add_argument("--beta_list", nargs="+", default=["inf"])
    args = parser.parse_args()

    # 1) Fine-tune the toxic model
    fine_tune_toxic_model(
        base_model_name=args.base_model_name,
        auth_token=args.auth_token,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        random_seed=args.random_seed_training
    )

    # 2) Generate detox text
    alpha_values = [float(x) for x in args.alpha_list]
    def parse_beta(x):
        return float(x) if x != "inf" else float("inf")
    beta_values = [parse_beta(x) for x in args.beta_list]

    generate_detox_text(
        base_model_name=args.base_model_name,
        toxic_model_ckpt=os.path.join(args.output_dir, "checkpoint-15126"),
        output_csv_dir=f"./distilled_text",
        alpha_list=alpha_values,
        beta_list=beta_values,
        random_seed=args.random_seed_distilling
    )

if __name__ == "__main__":
    main()
