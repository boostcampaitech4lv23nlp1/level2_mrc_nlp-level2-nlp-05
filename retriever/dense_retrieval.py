import argparse
import json
import os
import pickle
import time
from contextlib import contextmanager
from typing import Callable, Dict, List, NoReturn, Optional, Tuple

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, BertModel, BertPreTrainedModel


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

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        pooled_output = outputs[1]

        return pooled_output


class DenseRetrieval:
    def __init__(
        self,
        tokenizer: Callable[[str], List[str]],
        p_encoder_path: Optional[str],
        q_encoder_path: Optional[str],
        data_path: Optional[str] = "./dataset",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> None:

        self.tokenizer = tokenizer
        self.datasets = load_from_disk("./dataset/train_dataset/")

        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            self.wiki = json.load(f)

        self.p_encoder = BertEncoder.from_pretrained(p_encoder_path)
        self.q_encoder = BertEncoder.from_pretrained(q_encoder_path)
        if torch.cuda.is_available():
            self.p_encoder.cuda()
            self.q_encoder.cuda()

        dense_embedding_path = data_path + "dense_embedding.bin"

        if os.path.isfile(dense_embedding_path):
            with open(dense_embedding_path, "rb") as f:
                self.p_embs = pickle.load(f)
        else:
            self.p_embs = self.get_wiki_dense_embedding(self.p_encoder)
            with open(dense_embedding_path, "wb") as f:
                pickle.dump(self.p_embs, f)

    def get_wiki_dense_embedding(self, p_encoder):
        wiki_corpus = list(
            set([self.wiki[str(i)]["text"] for i in range(len(self.wiki))])
        )
        eval_batch_size = 32
        p_seqs = self.tokenizer(
            wiki_corpus,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        dataset = TensorDataset(
            p_seqs["input_ids"], p_seqs["attention_mask"], p_seqs["token_type_ids"]
        )
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=eval_batch_size)

        p_embs = []

        with torch.no_grad():
            self.p_encoder.eval()

            epoch_iterator = tqdm(dataloader, desc="Iteration")
            p_embs = []
            for _, batch in enumerate(epoch_iterator):
                batch = tuple(t.cuda() for t in batch)
                p_inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }
                outputs = p_encoder(**p_inputs).to("cpu").numpy()
                p_embs.extend(outputs)

        torch.cuda.empty_cache()
        p_embs = np.array(p_embs)

        return p_embs

    def get_topk_doc_id_and_score(self, query, top_k):
        with torch.no_grad():
            self.q_encoder.eval()

            q_seqs_val = self.tokenizer(
                [query], padding="max_length", truncation=True, return_tensors="pt"
            ).to("cuda")
            q_emb = self.q_encoder(**q_seqs_val).to("cpu")  # (num_query, emb_dim)

            p_embs = torch.Tensor(self.p_embs).squeeze()  # (num_passage, emb_dim)
            dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))

            rank = (
                torch.argsort(dot_prod_scores, dim=1, descending=True)
                .squeeze()
                .to("cpu")
                .numpy()
                .tolist()
            )
            scores = []
            for r in rank[:top_k]:
                scores.append(dot_prod_scores[0][r].item())

        return rank[:top_k], scores


if __name__ == "__main__":
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    args, _ = parser.parse_known_args()
    cfg = OmegaConf.load(f"./config/{args.config}/dense_config.yaml")

    data_path = "./dataset/"
    context_path = "wikipedia_documents.json"
    data_args = cfg.data_args
    training_args = cfg.training_args
    model_args = cfg.model_args

    model_name = model_args.dense_train_model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    p_encoder_path = model_args.p_encoder_path
    q_encoder_path = model_args.q_encoder_path

    retrieval = DenseRetrieval(
        tokenizer=tokenizer,
        p_encoder_path=p_encoder_path,
        q_encoder_path=q_encoder_path,
        data_path=data_path,
        context_path=context_path,
    )

    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

    print(retrieval.get_topk_doc_id_and_score(query, 100))
