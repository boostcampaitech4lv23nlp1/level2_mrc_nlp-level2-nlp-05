import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.utils import logging


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

        data_path = "./dataset/"
        context_path = "wikipedia_documents.json"

        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            self.wiki = json.load(f)

        self.wiki_corpus = list(
            set([self.wiki[str(i)]["text"] for i in range(len(self.wiki))])
        )

        wiki_iterator = tqdm(self.wiki_corpus, desc="Iteration")
        self.wiki_tokens = []

        print("Wiki documents Tokeniation")
        for p in wiki_iterator:
            token = self.tokenizer(
                p, padding="max_length", truncation=True, return_tensors="pt"
            ).to("cuda")
            self.wiki_tokens.append(token)

    # TODO : Set optimizer
    def set_optimizer():
        pass

    def train_per_epoch(self, epoch_iterator: DataLoader, optimizer, scheduler):
        # loss 계산
        batch_loss = 0
        train_acc = 0

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

            wiki_iterator = tqdm(self.wiki_tokens, desc="Iteration")
            p_embs = []
            for p in wiki_iterator:
                p_emb = self.p_encoder(**p).to("cpu").numpy()
                p_embs.append(p_emb)

            p_embs = torch.Tensor(p_embs).squeeze()  # (num_passage, emb_dim)

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
            {
                "params": [
                    p
                    for n, p in self.p_encoder.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.p_encoder.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p
                    for n, p in self.q_encoder.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.q_encoder.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            eps=self.args.adam_epsilon,
        )
        t_total = (
            len(train_dataloader)
            // self.args.gradient_accumulation_steps
            * self.args.num_train_epochs
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=t_total,
        )

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
                    self.p_encoder.save_pretrained("./retriever/saved_models/p_encoder")
                    self.p_encoder.save_pretrained("./retriever/saved_models/q_encoder")

                    best_score = top_100

        return self.p_encoder, self.q_encoder
