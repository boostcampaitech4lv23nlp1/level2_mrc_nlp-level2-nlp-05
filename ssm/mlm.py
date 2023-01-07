from datasets import load_dataset, load_from_disk

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoConfig,
    DataCollatorForLanguageModeling,
)
import torch
from transformers import Trainer, TrainingArguments

import argparse
from omegaconf import OmegaConf

import random
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from random import randint
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import os
import json

def load_data():
    dataset_path = "../dataset/wikipedia_documents.json"
    with open(dataset_path, "r") as f:
        wiki = json.load(f)

    wiki_texts= list(dict.fromkeys([v["text"] for v in wiki.values()]))    
    wiki_corpus = [{"document_text": wiki_texts[i]} for i in range(len(wiki_texts))]
    return wiki_corpus


def make_data():
    with open("../dataset/wikipedia_documents2.json", "w") as f:
        json.dump(load_data(), f)


def load_mlm_dataset(tokenizer, datasets):
    # preprocessing
    def prepare_validation_features(examples):
        # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
        # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
        tokenized_examples = tokenizer(
            examples['document_text'],
            max_length=384,
            stride=128,
            truncation=True,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=False if 'roberta' in tokenizer.name_or_path else True, # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
            padding="max_length",
        )
        
        tokenized_examples.pop('offset_mapping')
        tokenized_examples.pop('overflow_to_sample_mapping')
        
        return tokenized_examples

    train_dataset = datasets["train"]

    # Validation Feature 생성
    train_dataset = train_dataset.map(
        prepare_validation_features,
        batched=True,  # defalut batch size는 1000입니다. 
        remove_columns=["document_text"]
    )
    return train_dataset


def tapt_pretrain():
    model_name = 'klue/roberta-large'

    # set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    #RE_dataset = dataloader.load_predict_dataset(tokenizer, conf.path.pretrain_path, conf)
    if not os.path.exists('../dataset/wikipedia_documents2.json'):
        print("make data")
        make_data()
    dataset = load_dataset("json", data_files="../dataset/wikipedia_documents2.json")
    train_dataset = load_mlm_dataset(tokenizer, dataset)
    
    # Pretrained model for MaskedLM training
    model_config = AutoConfig.from_pretrained(model_name)  # 모델 가중치 불러오기
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForMaskedLM.from_pretrained(model_name, config=model_config)
    model.resize_token_embeddings(len(tokenizer))
    model.parameters
    model.to(device)

    # token 15% 확률 masking 진행
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # TAPT task이기 때문에 evaluation_strategy X
    # cuda out-of-memory 발생하여 fp16 = True 로 변경
    training_args = TrainingArguments(
        output_dir="./klue-roberta-pretrained_mlm",
        learning_rate=3e-05,
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        save_steps=4000,
        save_total_limit=2,
        save_strategy="steps",
        logging_dir="./logs",
        logging_steps=100,
        fp16=True, # 16비트로 변환
        fp16_opt_level="O1",
        resume_from_checkpoint=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    model.save_pretrained("./klue-roberta-pretrained_mlm")  # pretrained_model save

if __name__ == "__main__":
    
    tapt_pretrain()