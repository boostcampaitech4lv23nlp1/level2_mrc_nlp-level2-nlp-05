import argparse
import json
import os
import pickle
import time
from contextlib import contextmanager
from typing import Callable, Dict, List, NoReturn, Optional, Tuple, Union

import faiss
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_from_disk, concatenate_datasets
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, BertModel, BertPreTrainedModel
from dense_model import BertEncoder

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

class DenseRetrieval:
    def __init__(
        self,
        tokenizer: Callable[[str], List[str]],
        indexer: Optional[str],
        p_encoder_path: Optional[str],
        q_encoder_path: Optional[str],
        data_path: Optional[str] = "../dataset",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> None:

        self.tokenizer = tokenizer
        self.datasets = load_from_disk("../dataset/train_dataset/")

        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            self.wiki = json.load(f)

        # dense_trainer와 똑같이 설정
        self.contexts = list(
            set([self.wiki[str(i)]["text"] for i in range(len(self.wiki))])
        )
        
        self.ids = list(range(len(self.contexts)))
        
        self.indexer = faiss.read_index(indexer)
        self.p_encoder = BertEncoder.from_pretrained(p_encoder_path)
        self.q_encoder = BertEncoder.from_pretrained(q_encoder_path)
        if torch.cuda.is_available():
            self.p_encoder.cuda()
            self.q_encoder.cuda()

        
        dense_embedding_path = data_path + "dense_embedding.bin"

        # for kodpr
        # dense_embedding_path = data_path + "dense_embedding(kodpr).bin"

        if os.path.isfile(dense_embedding_path):
            with open(dense_embedding_path, "rb") as f:
                self.p_embs = pickle.load(f)
        else:
            self.p_embs = self.get_wiki_dense_embedding(self.p_encoder)
            with open(dense_embedding_path, "wb") as f:
                pickle.dump(self.p_embs, f)

        self.device = "cuda"

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

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.
        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]
        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
            retrieve와 같은 기능을 하지만 faiss.indexer를 사용합니다.
        """
        # assert self.indexer is not None, "indexer가 없습니다."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query, k=topk)
        
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            queries = query_or_dataset["question"]
            total = []

            with timer("query bulk search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    queries=queries, k=topk
                )

            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Dense retrieval: ")
            ) :
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    # 여기 좀 이상한데 고쳐야 할 것 같은데
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            result = pd.DataFrame(total)
            return result

    def retrieve_faiss(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.
        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]
        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
            retrieve와 같은 기능을 하지만 faiss.indexer를 사용합니다.
        """

        assert self.indexer is not None, "indexer가 없습니다."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc_faiss(
                query_or_dataset, k=topk
            )
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            queries = query_or_dataset["question"]
            total = []

            with timer("query faiss search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk_faiss(
                    queries, k=topk
                )

            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Dense faiss retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    # 여기 좀 이상한데 고쳐야 할 것 같은데
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            result = pd.DataFrame(total)
            return result

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """
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
        for r in rank[:k]:
            scores.append(dot_prod_scores[0][r].item())

        return scores, rank[:k] 

    def get_relevant_doc_bulk(self, queries: List, k: Optional[int] = 1, batch_size: Optional[int] = 16
    ) -> Tuple[List, List]:
        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """
        # Batch tokenizer
        q_tokenized = self.tokenizer(
            queries, padding="max_length", truncation=True, return_tensors="pt"
        )
        queries_dataset = TensorDataset(
            q_tokenized["input_ids"],
            q_tokenized["attention_mask"],
            q_tokenized["token_type_ids"],
        )
        queries_dataloader = DataLoader(queries_dataset, batch_size=batch_size)

        # Get dense vector(query encoder hidden size) of bulk query
        with torch.no_grad():
            self.q_encoder.eval()
            q_embs = []
            for batch in tqdm(queries_dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                q_inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }
                # q_emb.shape = (batch_size, hidden_size)
                q_emb = self.q_encoder(**q_inputs).to("cpu")
                q_embs.append(q_emb)
        
        # output.shape = (num_passage, hidden_size)
        q_embs = torch.cat(q_embs, dim=0).view(len(queries_dataset), -1)

        p_embs = torch.Tensor(self.p_embs).squeeze()  # (num_passage, emb_dim)
        dot_prod_scores = torch.matmul(q_embs, torch.transpose(p_embs, 0, 1))

        ranks = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()

        scores_list = []
        ranks_list = []

        for i in range(len(ranks)) : 
            scores = []
            for r in ranks[i][:k]:
                scores.append(dot_prod_scores[i][r].item())
            
            scores_list.append(scores)
            ranks_list.append(ranks[i][:k])

        return scores_list, ranks_list 

    def get_relevant_doc_faiss(
        self, query: str, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        # Tokenize
        q_tokenized = self.tokenizer(
            query, padding="max_length", truncation=True, return_tensors="pt"
        )

        # Get dense vector(query encoder hidden size) of a single query
        with torch.no_grad():
            self.q_encoder.eval()
            q_tokenized.to(self.device)
            output = self.q_encoder(**q_tokenized).to(self.device)

        # output.shape = (1,hidden_size)
        output = np.float32(output.to("cpu"))

        # Search in indexer
        with timer("query faiss search"):
            D, I = self.indexer.search(output, k)

        return D.tolist()[0], I.tolist()[0]

    def get_relevant_doc_bulk_faiss(
        self, queries: List, k: Optional[int] = 1, batch_size: Optional[int] = 16
    ) -> Tuple[List, List]:

        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """
        # Batch tokenizer
        q_tokenized = self.tokenizer(
            queries, padding="max_length", truncation=True, return_tensors="pt"
        )
        queries_dataset = TensorDataset(
            q_tokenized["input_ids"],
            q_tokenized["attention_mask"],
            q_tokenized["token_type_ids"],
        )
        queries_dataloader = DataLoader(queries_dataset, batch_size=batch_size)

        # Get dense vector(query encoder hidden size) of bulk query
        with torch.no_grad():
            self.q_encoder.eval()
            q_embs = []
            for batch in tqdm(queries_dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                q_inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }
                # q_emb.shape = (batch_size, hidden_size)
                q_emb = self.q_encoder(**q_inputs).to(self.device)
                q_embs.append(q_emb)

        # output.shape = (num_passage, hidden_size)
        q_embs = torch.cat(q_embs, dim=0).view(len(queries_dataset), -1)
        # Convert torch_tensor into np.float32
        q_embs = np.float32(q_embs.to("cpu"))

        D, I = self.indexer.search(q_embs, k)
        return D.tolist(), I.tolist()



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
        indexer='', # 추가 예정
        p_encoder_path=p_encoder_path,
        q_encoder_path=q_encoder_path,
        data_path=data_path,
        context_path=context_path,
    )

    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

    org_dataset = load_from_disk("./dataset/train_dataset")
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )

    with timer("bulk query by normal search"):
        df = retrieval.retrieve(query_or_dataset=full_ds, topk=1)
        correct_list = []
        for i in range(len(df)) :
            isInContext = df["original_context"].iloc[i] in df["context"].iloc[i]
            correct_list.append(isInContext)

        print("correct retrieval result by DPR", sum(correct_list) / len(df))
    
    print(len(df.iloc[0]['context']))
    
    with timer("bulk query by faiss search"):
        df = retrieval.retrieve_faiss(query_or_dataset=full_ds, topk=1)
        correct_list = []
        for i in range(len(df)) :
            isInContext = df["original_context"].iloc[i] in df["context"].iloc[i]
            correct_list.append(isInContext)

        print("correct retrieval result by DPR", sum(correct_list) / len(df))

