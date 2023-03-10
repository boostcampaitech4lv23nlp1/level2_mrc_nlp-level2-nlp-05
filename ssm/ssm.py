from datasets import load_dataset, load_from_disk

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoConfig,
    DataCollatorForLanguageModeling,
)
import torch
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding

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
from collections import defaultdict

class MyDataCollatorWithPadding(DataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        max_len = 0
        for i in features:
            if len(i["input_ids"]) > max_len:
                max_len = len(i["input_ids"])
        
        batch = defaultdict(list)
        for item in features:
            for k in item:
                item[k] = torch.Tensor(item[k])
                padding_len = max_len - item[k].size(0)
                if k == "input_ids":
                    item[k] = torch.cat((item[k], torch.tensor([self.tokenizer.pad_token_id] * padding_len)), dim=0)
                else:
                    item[k] = torch.cat((item[k], torch.tensor([0] * padding_len)), dim=0)
                batch[k].append(item[k])

        for k in batch:
            batch[k] = torch.stack(batch[k], dim=0)
            batch[k] = batch[k].to(torch.long)
        
        return batch

def load_data():  # 위키피디아 데이터에서 entity를 찾아 마스킹합니다.
    dataset_path = "../dataset/wikipedia_documents.json"
    with open(dataset_path, "r") as f:
        wiki = json.load(f)

    wiki_texts_prev = list(dict.fromkeys([v["text"] for v in wiki.values()]))
    
    wiki_texts = []
    for t in wiki_texts_prev:
        while(len(t) > 500):
            wiki_texts.append(t[:500])
            t = t[384:]  # stride = 500 - 384 
    
    wiki_strings = []
    wiki_labels = []
    wiki_inputs = []
    for text in tqdm(wiki_texts):
        string = ''
        labels = []
        inputs = []
        for pred in ner(text):
            if pred[1] != 'O' and np.random.normal(mu, sigma) < 0:
                mask = tokenizer(pred[0])['input_ids'][1:-1]
                ln = len(mask)
                labels += mask
                inputs += tokenizer('[MASK]' * ln)['input_ids'][1:-1]
                string += '[MASK]' * ln

            else:
                labels += tokenizer(pred[0])['input_ids'][1:-1]
                inputs += tokenizer(pred[0])['input_ids'][1:-1]
                string += pred[0]
        wiki_labels.append(labels)
        wiki_inputs.append(inputs)
        wiki_strings.append(string)
            
    wiki_corpus = [{"document_text": wiki_texts[i], "masked_strings": wiki_strings[i], "inputs":wiki_inputs[i], "labels":wiki_labels[i]} for i in range(len(wiki_texts))]
    return wiki_corpus

def make_data():
    with open("../dataset/wikipedia_documents_ssm.json", "w") as f:
        json.dump(load_data(), f)

        
def load_ssm_dataset(tokenizer, datasets):
    # preprocessing
    def prepare_validation_features(examples):
        # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
        # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
        
        final_examples = {}
        final_examples['input_ids'] = examples['inputs']
        final_examples['labels'] = examples['labels']
        
        final_examples['attention_mask'] = []
        for i in examples['inputs']:
            final_examples['attention_mask'].append([1]*len(i))
        
        return final_examples

    train_dataset = datasets["train"]

    # Validation Feature 생성
    train_dataset = train_dataset.map(
        prepare_validation_features,
        batched = True,
        remove_columns=['document_text', 'masked_strings', 'inputs', 'labels']
    )
    return train_dataset


def ssm_pretrain():
    model_name = 'klue/roberta-large'

    # set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if not os.path.exists('../dataset/wikipedia_documents_ssm.json'):
        print("make data")
        make_data()
    dataset = load_dataset("json", data_files="../dataset/wikipedia_documents_ssm_qa2.json")
    train_dataset = load_ssm_dataset(tokenizer, dataset)

    # Pretrained model for MaskedLM training
    model_config = AutoConfig.from_pretrained(model_name)  # 모델 가중치 불러오기
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForMaskedLM.from_pretrained(model_name, config=model_config)
    #model = AutoModelForMaskedLM.from_pretrained('klue-roberta-pretrained3')
    model.resize_token_embeddings(len(tokenizer))
    model.parameters
    model.to(device)

    data_collator = MyDataCollatorWithPadding(tokenizer)

    # cuda out-of-memory 발생하여 fp16 = True 로 변경
    training_args = TrainingArguments(
        output_dir="./klue-roberta-pretrained_qa2",
        learning_rate=3e-05,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        save_steps=5000,
        save_total_limit=1,
        save_strategy="steps",
        logging_dir="./logs",
        logging_steps=200,
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
    model.save_pretrained("./klue-roberta-pretrained_qa2")  # pretrained_model save

if __name__ == "__main__":
    ssm_pretrain()