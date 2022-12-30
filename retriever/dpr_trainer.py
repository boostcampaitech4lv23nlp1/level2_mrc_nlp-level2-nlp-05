import json
import random
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pprint import pprint

from sklearn.feature_extraction.text import TfidfVectorizer

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    BertModel, BertPreTrainedModel,
    AdamW, get_linear_schedule_with_warmup,
    TrainingArguments,
    HfArgumentParser
)

from datasets import DatasetDict, load_from_disk, load_metric
from omegaconf import OmegaConf
from datetime import datetime
from config.dpr_arguments import DprArguments
import faiss

# 난수 고정
def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    random.seed(random_seed)
    np.random.seed(random_seed)
    
set_seed(42) # magic number :)

class BertEncoder(BertPreTrainedModel):

    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()
      
      
    def forward(
            self,
            input_ids, 
            attention_mask=None,
            token_type_ids=None
        ): 
  
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        pooled_output = outputs[1]
        return pooled_output

class DenseTrainer:

    def __init__(self, args, dataset, num_neg, tokenizer, p_encoder, q_encoder):

        self.args = args
        self.dataset = dataset
        self.num_neg = num_neg

        self.tokenizer = tokenizer
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder
        self.passage_dataloader = None

        now = datetime.now()
        self.current_time = now.strftime("%d-%H-%M")

        self.prepare_in_batch_negative(num_neg=self.num_neg)

    def parse_wiki_data(self):

        print("build wiki data")
        with open('../dataset/wikipedia_documents.json', "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로

        valid_seqs = self.tokenizer(self.contexts, padding="max_length", truncation=True, return_tensors='pt')
        passage_dataset = TensorDataset(
            valid_seqs['input_ids'], valid_seqs['attention_mask'], valid_seqs['token_type_ids']
        )
        # 이거를 위키로 바꿔야 하고
        self.passage_dataloader = DataLoader(passage_dataset, batch_size=self.args.per_device_train_batch_size)

    def prepare_in_batch_negative(self, dataset=None, num_neg=2, tokenizer=None):

        print("## preparing in_batch_negative ##")
        print("num_neg :",num_neg)

        if dataset is None:
            dataset = self.dataset

        if tokenizer is None:
            tokenizer = self.tokenizer

        # 1. In-Batch-Negative 만들기
        # CORPUS를 np.array로 변환해줍니다.        
        corpus = np.array(list(set([example for example in dataset['context']])))
        p_with_neg = []

        for c in dataset['context']:
            while True:
                neg_idxs = np.random.randint(len(corpus), size=num_neg)

                if not c in corpus[neg_idxs]:
                    p_neg = corpus[neg_idxs]

                    p_with_neg.append(c)
                    p_with_neg.extend(p_neg)
                    break

        #######################
        # 여기에 bm25 들어가야함
        #######################

        # 2. (Question, Passage) 데이터셋 만들어주기
        q_seqs = tokenizer(dataset['question'], padding="max_length", truncation=True, return_tensors='pt')
        p_seqs = tokenizer(p_with_neg, padding="max_length", truncation=True, return_tensors='pt')

        max_len = p_seqs['input_ids'].size(-1)
        p_seqs['input_ids'] = p_seqs['input_ids'].view(-1, num_neg+1, max_len)
        p_seqs['attention_mask'] = p_seqs['attention_mask'].view(-1, num_neg+1, max_len)
        p_seqs['token_type_ids'] = p_seqs['token_type_ids'].view(-1, num_neg+1, max_len)

        train_dataset = TensorDataset(
            p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'], 
            q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids']
        )

        self.train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.args.per_device_train_batch_size)

    def train(self, args=None):

        if args is None:
            args = self.args
        batch_size = args.per_device_train_batch_size

        # Optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.p_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in self.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.q_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        t_total = len(self.train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

        # Start training!
        global_step = 0

        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = tqdm(range(int(args.num_train_epochs)), desc="Epoch")
        # for _ in range(int(args.num_train_epochs)):
        for _ in train_iterator:

            with tqdm(self.train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:

                    self.p_encoder.train()
                    self.q_encoder.train()
            
                    targets = torch.zeros(batch_size).long() # positive example은 전부 첫 번째에 위치하므로
                    targets = targets.to(args.device)

                    p_inputs = {
                        'input_ids': batch[0].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                        'attention_mask': batch[1].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                        'token_type_ids': batch[2].view(batch_size * (self.num_neg + 1), -1).to(args.device)
                    }
            
                    q_inputs = {
                        'input_ids': batch[3].to(args.device),
                        'attention_mask': batch[4].to(args.device),
                        'token_type_ids': batch[5].to(args.device)
                    }
            
                    p_outputs = self.p_encoder(**p_inputs)  # (batch_size*(num_neg+1), emb_dim)
                    q_outputs = self.q_encoder(**q_inputs)  # (batch_size*, emb_dim)

                    # Calculate similarity score & loss
                    p_outputs = p_outputs.view(batch_size, self.num_neg + 1, -1)
                    q_outputs = q_outputs.view(batch_size, 1, -1)

                    sim_scores = torch.bmm(q_outputs, torch.transpose(p_outputs, 1, 2)).squeeze()  #(batch_size, num_neg + 1)
                    sim_scores = sim_scores.view(batch_size, -1)
                    sim_scores = F.log_softmax(sim_scores, dim=1)

                    loss = F.nll_loss(sim_scores, targets)
                    tepoch.set_postfix(loss=f'{str(loss.item())}')

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    self.p_encoder.zero_grad()
                    self.q_encoder.zero_grad()

                    global_step += 1

                    torch.cuda.empty_cache()

                    del p_inputs, q_inputs
    
    def save_model(self):

        torch.save(self.p_encoder, f'./dpr_model/p_encoder_{self.current_time}.pt')
        torch.save(self.q_encoder, f'./dpr_model/q_encoder_{self.current_time}.pt')


    def build_faiss(self, p_encoder=None, num_clusters=64):

        args = self.args

        # Load passage encoder
        p_encoder = self.p_encoder if p_encoder == None else torch.load(p_encoder)

        # Load and parse wiki_documents.json
        if self.passage_dataloader == None: self.parse_wiki_data()

        # Convert wiki contexts into dense matrix
        with torch.no_grad():
            p_encoder.eval()

            p_embs = []
            for batch in tqdm(self.passage_dataloader):

                batch = tuple(t.to(args.device) for t in batch)
                p_inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2]
                }
                p_emb = p_encoder(**p_inputs).to(args.device)
                p_embs.append(p_emb)

        # 이거 torch.stack이 shape가 안맞아서 torch.cat으로 바꾸었는데 이상한지 잘 봐주세요
        p_embs = torch.cat(p_embs, dim=0).view(len(self.passage_dataloader.dataset), -1)  # (num_passage, emb_dim)
        p_embs = p_embs.to('cpu')
        # Convert dense matrix into faiss indexer
        indexer_name = f"faiss_clusters{num_clusters}_{self.current_time}.index"
        indexer_path = os.path.join('./dpr_model', indexer_name)

        p_emb = np.float32(p_embs)
        emb_dim = p_emb.shape[-1]

        num_clusters = num_clusters
        quantizer = faiss.IndexFlatL2(emb_dim)

        self.indexer = faiss.IndexIVFScalarQuantizer(
            quantizer, quantizer.d, num_clusters, faiss.METRIC_L2
        )
        self.indexer.train(p_emb)
        self.indexer.add(p_emb)
        faiss.write_index(self.indexer, indexer_path)
        print("Faiss Indexer Saved.")


def main(args):

    train_dataset = load_from_disk(args.train_dataset)
    train_dataset = train_dataset['train']

    model_checkpoint = args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    p_encoder = BertEncoder.from_pretrained(model_checkpoint).to(args.device)
    q_encoder = BertEncoder.from_pretrained(model_checkpoint).to(args.device)

    retriever = DenseTrainer(
                                args=args,
                                dataset=train_dataset,
                                num_neg=args.num_neg,
                                tokenizer=tokenizer,
                                p_encoder=p_encoder,
                                q_encoder=q_encoder)

    #retriever.train(args)

    #if args.model_save: retriever.save_model()
    if args.build_faiss: retriever.build_faiss('./dpr_model/p_encoder.pt')

if __name__ == "__main__":

    parser = HfArgumentParser((DprArguments))
    args = parser.parse_args_into_dataclasses()

    main(args[0])



    # q랑 p를 저장을 해야한다. => 저장은 된 것 같고
    # 세팅을 저장을 하고 ? 하이퍼 파라미터요

    # 이거를 indexing을 해가지고 faiss로 빠르게 불러와야한다.

    # retrieve 성능이 개쓰레기인데?

    # wandb로 로깅을 해야하나?

    # if mode == 'retrieve':
                    
    # model_checkpoint = cfg.model_args.model_name_or_path
    # tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    # p_encoder = torch.load('./dpr_model/p_encoder.pt').to('cuda')
    # q_encoder = torch.load('./dpr_model/q_encoder.pt').to('cuda')

    # retriever = DenseRetrieval(args=args, dataset=train_dataset, num_neg=cfg.train_args.num_neg, tokenizer=tokenizer, p_encoder=p_encoder, q_encoder=q_encoder)
    # query = '제주도의 시청의 주소는 뭐야?'
    
    # results = retriever.get_relevant_doc(query=query, k=cfg.rtv_args.top_k)
    # breakpoint()


# def get_relevant_doc(self, query, k=1, args=None, p_encoder=None, q_encoder=None):

#         if self.passage_dataloader == None: self.parse_wiki_data()

#         if args is None:
#             args = self.args

#         if p_encoder is None:
#             p_encoder = self.p_encoder

#         if q_encoder is None:
#             q_encoder = self.q_encoder

#         with torch.no_grad():
#             p_encoder.eval()
#             q_encoder.eval()

#             q_seqs_val = self.tokenizer([query], padding="max_length", truncation=True, return_tensors='pt').to(args.device)
#             q_emb = q_encoder(**q_seqs_val).to(args.device)  # (num_query=1, emb_dim)

#             p_embs = []
#             for batch in tqdm(self.passage_dataloader):

#                 batch = tuple(t.to(args.device) for t in batch)
#                 p_inputs = {
#                     'input_ids': batch[0],
#                     'attention_mask': batch[1],
#                     'token_type_ids': batch[2]
#                 }
#                 p_emb = p_encoder(**p_inputs).to(args.device)
#                 p_embs.append(p_emb)
                
#         # 이거 torch.stack이 shape가 안맞아서 torch.cat으로 바꾸었는데 이상한지 잘 봐주세요
#         p_embs = torch.cat(p_embs, dim=0).view(len(self.passage_dataloader.dataset), -1)  # (num_passage, emb_dim)

#         dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
#         rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
#         return rank[:k]