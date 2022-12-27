import time
import os
import json
from contextlib import contextmanager
from typing import Callable, Dict, List, NoReturn, Tuple, Optional

from tqdm import tqdm, trange
import argparse
from arguments import DataTrainingArguments, ModelArguments
import random
import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
from transformers import BertModel, BertPreTrainedModel, AdamW, TrainingArguments, get_linear_schedule_with_warmup

from datasets import DatasetDict, load_from_disk, load_metric

from transformers import AutoTokenizer
import numpy as np

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()
      
    def forward(self, input_ids, 
                attention_mask=None, token_type_ids=None): 
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
      
        pooled_output = outputs[1]
        
        return pooled_output        

class DenseRetrieval:
    def __init__(
        self,
        tokenizer: Callable[[str], List[str]],
        data_path: Optional[str] = "./dataset",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> None:
    
        self.tokeinzer = tokenizer
        self.datasets = load_from_disk("./dataset/train_dataset/")
        
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            self.wiki = json.load(f)
            
        # datasets에서 train 데이터 뽑아내기 
        training_dataset = self.datasets['train']
        
        # query와 passage를 위한 sequence를 tokenization
        q_seqs = tokenizer(training_dataset['question'], padding="max_length", truncation=True, return_tensors='pt')
        p_seqs = tokenizer(training_dataset['context'], padding="max_length", truncation=True, return_tensors='pt')

        
        # 데이터셋을 학습하기 위해서 TensorDataset으로 변경
        self.train_dataset = TensorDataset(p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'], 
                                      q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids'])

    
        
    def train(self, args, p_model, q_model) :
        # Dataloader
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=args.per_device_train_batch_size)
    

        # Optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in p_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in p_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in q_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in q_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
        
        # Start training!
        global_step = 0
        
        p_model.zero_grad()
        q_model.zero_grad()
        torch.cuda.empty_cache()
        
        train_iterator = tqdm(range(int(args.num_train_epochs)), desc="Epoch")
        
        # ## 이중 tqdm 어떻게 한 줄로??
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", leave=True)
            
            for step, batch in enumerate(epoch_iterator):
                p_model.train()
                q_model.train()
                
                if torch.cuda.is_available():
                    batch = tuple(t.cuda() for t in batch)
                
                # Tensor dataset 0, 1, 2 - passage
                p_inputs = {'input_ids': batch[0],
                            'attention_mask': batch[1],
                            'token_type_ids': batch[2]
                        }
                
                # Tensor dataset 3, 4, 5 - query
                q_inputs = {'input_ids': batch[3],
                            'attention_mask': batch[4],
                            'token_type_ids': batch[5]
                        }
                
                p_outputs = p_model(**p_inputs)  # (batch_size, emb_dim)
                q_outputs = q_model(**q_inputs)  # (batch_size, emb_dim)
                
                # Calculate similarity score & loss
                # (batch_size, emb_dim) x (emb_dim, batch_size) = (batch_size, batch_size)
                sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1))  
                
                # target: position of positive samples = diagonal element 
                targets = torch.arange(0, args.per_device_train_batch_size).long()
                if torch.cuda.is_available():
                    targets = targets.to('cuda')

                sim_scores = F.log_softmax(sim_scores, dim=1)
                
                loss = F.nll_loss(sim_scores, targets)
                # print('loss =', loss)
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                q_model.zero_grad()
                p_model.zero_grad()
                global_step += 1
                
                torch.cuda.empty_cache()


        return p_model, q_model
        
        
        

if __name__ == "__main__":
    from omegaconf import OmegaConf
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='')
    args, _ = parser.parse_known_args()
    cfg = OmegaConf.load(f'./config/{args.config}/base_config.yaml')
    
    model_name = cfg.model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    data_path = "./dataset/"
    context_path = "wikipedia_documents.json"
    data_args = cfg.data_args
    training_args = cfg.training_args
    model_args = cfg.model_args
    
    retrieval = DenseRetrieval(tokenizer=tokenizer, data_path=data_path, context_path=context_path)
    
    # 왜 코드가 꺼지는거임??
    # load pre-trained model on cuda (if available)
    p_encoder = BertEncoder.from_pretrained(model_name)
    q_encoder = BertEncoder.from_pretrained(model_name)
    
    if torch.cuda.is_available():
        p_encoder.cuda()
        q_encoder.cuda()
        print("GPU enabled")
    
    args = TrainingArguments(
            output_dir="dense_retireval",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=2,
            weight_decay=0.01
        )
    
    p_encoder, q_encoder = retrieval.train(args, p_encoder, q_encoder)
    # print(args)

    
