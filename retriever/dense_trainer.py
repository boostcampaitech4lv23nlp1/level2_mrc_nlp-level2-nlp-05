import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.utils import logging
import faiss
import numpy as np
from datetime import datetime
from dense_model import BertEncoder
import pickle

class DenseRetrievalTrainer:
    def __init__(
        self, args, cfg, tokenizer, p_encoder, q_encoder, train_dataset, valid_dataset
    ):
        self.args = args  # training_args
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

        data_path = "../dataset/"
        context_path = "wikipedia_documents.json"

        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            self.wiki = json.load(f)

        self.wiki_corpus = list(
                set([self.wiki[str(i)]["text"] for i in range(len(self.wiki))])
            )
        
        print("wiki document tokenizing")
        self.wiki_tokens = self.tokenizer(self.wiki_corpus, padding="max_length", truncation=True, return_tensors="pt")
        print("wiki document tokenizing done")


    # TODO : Set optimizer
    def set_optimizer():
        pass

    def train_per_epoch(self, epoch_iterator: DataLoader, optimizer, scheduler):
        # loss 계산
        batch_loss = 0
        train_acc = 0
        
        if self.cfg.training_args.in_batch_neg:

            for train_step, batch in enumerate(epoch_iterator):
                self.p_encoder.train()
                self.q_encoder.train()

                if torch.cuda.is_available():
                    batch = tuple(t.cuda() for t in batch)
                    targets = torch.zeros(self.args.per_device_train_batch_size).long()
                    targets = targets.cuda()
                    
                # Tensor dataset 0, 1, 2 - passage
                p_inputs = {
                    "input_ids": batch[0].view(self.args.per_device_train_batch_size * (self.cfg.training_args.num_neg + 1), -1),
                    "attention_mask": batch[1].view(self.args.per_device_train_batch_size * (self.cfg.training_args.num_neg + 1), -1),
                    "token_type_ids": batch[2].view(self.args.per_device_train_batch_size * (self.cfg.training_args.num_neg + 1), -1),
                }

                # Tensor dataset 3, 4, 5 - query
                q_inputs = {
                    "input_ids": batch[3],
                    "attention_mask": batch[4],
                    "token_type_ids": batch[5],
                }

                p_outputs = self.p_encoder(**p_inputs)  # (batch_size, emb_dim)
                q_outputs = self.q_encoder(**q_inputs)  # (batch_size, emb_dim)

                p_outputs = p_outputs.view(self.args.per_device_train_batch_size, (self.cfg.training_args.num_neg + 1), -1)
                q_outputs = q_outputs.view(self.args.per_device_train_batch_size,1,-1)

                sim_scores = torch.bmm(q_outputs, torch.transpose(p_outputs, 1, 2)).squeeze()  #(batch_size, num_neg + 1)
                sim_scores = sim_scores.view(self.args.per_device_train_batch_size, -1)
                sim_scores = F.log_softmax(sim_scores, dim=1)

                loss = F.nll_loss(sim_scores, targets)

                batch_loss += loss.item()

                _, preds = torch.max(sim_scores, 1)  # 예측값 뽑아내기
                train_acc += torch.sum(
                    preds.cpu() == targets.cpu() / self.args.per_device_train_batch_size
                )

                loss.backward()
                optimizer.step()
                scheduler.step()
                self.p_encoder.zero_grad()
                self.q_encoder.zero_grad()

                epoch_iterator.set_description("Loss %.04f step %d" % (loss, train_step))
                # lr = scheduler.optimizer.param_groups[0]['lr']
                # print(lr)

                del p_inputs, q_inputs

            torch.cuda.empty_cache()
            return batch_loss / len(epoch_iterator)


        elif self.cfg.training_args.hard_neg:

            for train_step, batch in enumerate(epoch_iterator):
                self.p_encoder.train()
                self.q_encoder.train()

                if torch.cuda.is_available():
                    batch = tuple(t.cuda() for t in batch)
                    targets = torch.zeros(self.args.per_device_train_batch_size).long()
                    targets = targets.cuda()

                # Tensor dataset 0, 1, 2 - passage
                p_inputs = {
                    "input_ids": batch[0].view(self.args.per_device_train_batch_size * (self.cfg.training_args.num_neg + 2), -1),
                    "attention_mask": batch[1].view(self.args.per_device_train_batch_size * (self.cfg.training_args.num_neg + 2), -1),
                    "token_type_ids": batch[2].view(self.args.per_device_train_batch_size * (self.cfg.training_args.num_neg + 2), -1),
                }

                # Tensor dataset 3, 4, 5 - query
                q_inputs = {
                    "input_ids": batch[3],
                    "attention_mask": batch[4],
                    "token_type_ids": batch[5],
                }

                p_outputs = self.p_encoder(**p_inputs)  # (batch_size, emb_dim)
                q_outputs = self.q_encoder(**q_inputs)  # (batch_size, emb_dim)

                p_outputs = p_outputs.view(self.args.per_device_train_batch_size, (self.cfg.training_args.num_neg + 2), -1)
                q_outputs = q_outputs.view(self.args.per_device_train_batch_size,1,-1)

                sim_scores = torch.bmm(q_outputs, torch.transpose(p_outputs, 1, 2)).squeeze()  #(batch_size, num_neg + 1)
                sim_scores = sim_scores.view(self.args.per_device_train_batch_size, -1)
                sim_scores = F.log_softmax(sim_scores, dim=1)
                
                loss = F.nll_loss(sim_scores, targets)

                batch_loss += loss.item()

                _, preds = torch.max(sim_scores, 1)  # 예측값 뽑아내기
                train_acc += torch.sum(
                    preds.cpu() == targets.cpu() / self.args.per_device_train_batch_size
                )

                loss.backward()
                optimizer.step()
                scheduler.step()
                self.p_encoder.zero_grad()
                self.q_encoder.zero_grad()
                
                epoch_iterator.set_description("Loss %.04f step %d" % (loss, train_step))

                del p_inputs, q_inputs
                
            torch.cuda.empty_cache()
            return batch_loss / len(epoch_iterator)

        
        else:
            for train_step, batch in enumerate(epoch_iterator):
                self.p_encoder.train()
                self.q_encoder.train()

                if torch.cuda.is_available():
                    batch = tuple(t.cuda() for t in batch)

                # Tensor dataset 0, 1, 2 - passage
                p_inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }

                # Tensor dataset 3, 4, 5 - query
                q_inputs = {
                    "input_ids": batch[3],
                    "attention_mask": batch[4],
                    "token_type_ids": batch[5],
                }

                p_outputs = self.p_encoder(**p_inputs)  # (batch_size, emb_dim)
                q_outputs = self.q_encoder(**q_inputs)  # (batch_size, emb_dim)

                # Calculate similarity score & loss
                # (batch_size, emb_dim) x (emb_dim, batch_size) = (batch_size, batch_size)
                sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1))

                # target: position of positive samples = diagonal element
                targets = torch.arange(0, self.args.per_device_train_batch_size).long()
                if torch.cuda.is_available():
                    targets = targets.to("cuda")

                sim_scores = F.log_softmax(sim_scores, dim=1)

                loss = F.nll_loss(sim_scores, targets)
                batch_loss += loss.item()

                _, preds = torch.max(sim_scores, 1)  # 예측값 뽑아내기
                train_acc += torch.sum(
                    preds.cpu() == targets.cpu() / self.args.per_device_train_batch_size
                )

                loss.backward()
                optimizer.step()
                scheduler.step()
                self.p_encoder.zero_grad()
                self.q_encoder.zero_grad()

                del p_inputs, q_inputs

            torch.cuda.empty_cache()
            return batch_loss / len(epoch_iterator)

    # TODO : dev 만들기
    def dev_per_epoch(self, epoch_iterator: DataLoader):
        pass

    def evaluation(self):
        with torch.no_grad():
            self.p_encoder.eval()
            self.q_encoder.eval()

            query = self.valid_dataset["question"]
            ground_truth = self.valid_dataset["context"]

            q_seqs_val = self.tokenizer(
                query, padding="max_length", truncation=True, return_tensors="pt"
            ).to("cuda")
            q_emb = self.q_encoder(**q_seqs_val).to("cpu")  # (num_query, emb_dim)

            wiki_iterator = TensorDataset(
                self.wiki_tokens["input_ids"], self.wiki_tokens["attention_mask"], self.wiki_tokens["token_type_ids"],
            )
            wiki_dataloader = DataLoader(wiki_iterator, batch_size=64)
            
            p_embs = []
            for p in tqdm(wiki_dataloader):

                if torch.cuda.is_available():
                    p = tuple(t.cuda() for t in p)
                
                p_inputs = {
                    "input_ids": p[0],
                    "attention_mask": p[1],
                    "token_type_ids": p[2],
                }

                p_emb = self.p_encoder(**p_inputs)
                p_embs.append(p_emb)
            
            p_embs = torch.cat(p_embs, dim=0).view(len(wiki_iterator), -1)
            p_embs = p_embs.to('cpu') # (num_passage, emb_dim)

            dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))

            rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()

            def validation_score(k):
                total = len(query)
                correct_cnt = 0

                for i in range(total):
                    top_k = rank[i][:k]

                    pred_corpus = []
                    for top in top_k:
                        pred_corpus.append(self.wiki_corpus[top])

                    if ground_truth[i] in pred_corpus:
                        correct_cnt += 1

                result = correct_cnt / total
                print(f"Top-{k} score is {result:.4f}")
                return result

            print("========================= Evaluation =========================")

            return [
                validation_score(1),
                validation_score(5),
                validation_score(10),
                validation_score(30),
                validation_score(50),
                validation_score(100),
            ]

    def train(
        self,
    ):
        # Dataloader
        train_random_sampler = RandomSampler(self.train_dataset)
        # valid_random_sampler = RandomSampler(self.valid_dataset)

        train_dataloader = DataLoader(
            self.train_dataset,
            sampler=train_random_sampler,
            batch_size=self.args.per_device_train_batch_size,
        )
        # valid_dataloader = DataLoader(self.valid_dataset, sampler=valid_random_sampler, batch_size=self.args.per_device_eval_batch_size)

        # Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.p_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in self.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.q_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        t_total = (len(train_dataloader) * self.args.num_train_epochs)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=t_total*self.args.warmup_ratio, num_training_steps=t_total)

        # Start training!
        global_step = 0
        if torch.cuda.is_available():
            self.p_encoder.cuda()
            self.q_encoder.cuda()
            print("GPU enabled")

        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = tqdm(range(int(self.args.num_train_epochs)), desc="Epoch")
        best_score = 0

        for epoch in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", leave=True)
            
            # train per epoch
            train_loss = self.train_per_epoch(
                epoch_iterator=epoch_iterator, optimizer=optimizer, scheduler=scheduler
            )

            print(f"Train_loss : {train_loss:.4f}\n")

            if (epoch + 1) % 5 == 0:
                top_1, top_5, top_10, top_30, top_50, top_100 = self.evaluation()

                if top_100 > best_score:
                    self.p_encoder.save_pretrained("./retriever/saved_models/p_encoder_{cfg.training_args.p_encoder_save_name}")
                    self.q_encoder.save_pretrained("./retriever/saved_models/q_encoder_{cfg.training_args.q_encoder_save_name}")

                    best_score = top_100

        return self.p_encoder, self.q_encoder

    def build_faiss(self, p_encoder_path=None, num_clusters=64):
        now = datetime.now()
        current_time = now.strftime("%d-%H-%M")

        # Load passage encoder
        p_encoder = self.p_encoder if p_encoder_path == None else BertEncoder.from_pretrained(p_encoder_path)

        if torch.cuda.is_available():
            p_encoder.cuda()

        # Convert wiki contexts into dense matrix
        with torch.no_grad():
            p_encoder.eval()

            wiki_iterator = tqdm(self.wiki_tokens, desc="Iteration")
            p_embs = []
            # for p in wiki_iterator:
            for idx, p in enumerate(wiki_iterator) :
                if idx == 1000 :
                    break
                p_emb = p_encoder(**p).to("cpu").numpy()
                p_embs.append(p_emb)

            p_embs = torch.Tensor(p_embs).squeeze()  # (num_passage, emb_dim)

        # Convert dense matrix into faiss indexer
        indexer_name = f"faiss_clusters{num_clusters}_{current_time}.index"
        indexer_path = os.path.join('./dpr_model', indexer_name)

        p_emb = np.float32(p_embs)
        emb_dim = p_emb.shape[-1]

        num_clusters = num_clusters
        quantizer = faiss.IndexFlatL2(emb_dim)

        indexer = faiss.IndexIVFScalarQuantizer(
            quantizer, quantizer.d, num_clusters, faiss.METRIC_L2
        )
        indexer.train(p_emb)
        indexer.add(p_emb)
        faiss.write_index(indexer, indexer_path) # 여기서 왜 에러나는 거임?
        print("Faiss Indexer Saved.")


def main(cfg):
    from transformers import AutoTokenizer, TrainingArguments
    from dense_utils import BaseDataset
    from datasets import load_from_disk

    data_path = "./dataset/"
    context_path = "wikipedia_documents.json"
    data_args = cfg.data_args
    train_args = cfg.training_args
    model_args = cfg.model_args

    model_name = model_args.dense_train_model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_path = data_args.train_dataset_name
    valid_path = data_args.valid_dataset_name

    train_dataset = BaseDataset(tokenizer=tokenizer, datapath=valid_path) # 수정
    valid_dataset = load_from_disk(dataset_path=valid_path)

    print("Train Dataset Length:", len(train_dataset))
    print("Valid Dataset Length:", len(valid_dataset))

    p_encoder = BertEncoder.from_pretrained(model_name)
    q_encoder = BertEncoder.from_pretrained(model_name)

    # yaml
    args = TrainingArguments(
        output_dir="./dense_retireval",
        evaluation_strategy="epoch",
        learning_rate=train_args.learning_rate,
        per_device_train_batch_size=train_args.per_device_train_batch_size,
        per_device_eval_batch_size=train_args.per_device_eval_batch_size,
        num_train_epochs=train_args.num_train_epochs,
        weight_decay=train_args.weight_decay,
        gradient_accumulation_steps=train_args.gradient_accumulation_steps,
    )

    # Dense Retrieval Trainer
    DR_trainer = DenseRetrievalTrainer(
        args=args,
        cfg=cfg,
        tokenizer=tokenizer,
        p_encoder=p_encoder,
        q_encoder=q_encoder,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    )

    DR_trainer.build_faiss(p_encoder_path='./retriever/saved_models/p_encoder', num_clusters=64)


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    args, _ = parser.parse_known_args()
    cfg = OmegaConf.load(f"./config/{args.config}/dense_config.yaml")

    main(cfg)

    