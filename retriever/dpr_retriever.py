import json
import os
import pickle
import time
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union

import faiss
import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader, TensorDataset
from dpr_trainer import BertEncoder


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class DenseRetriever:
    def __init__(
        self,
        tokenize_fn,
        indexer,
        q_encoder,
        context_path: Optional[str] = "../dataset/wikipedia_documents.json",
    ) -> NoReturn:

        """
        Arguments:
            tokenize_fn:
                기본 text를 tokenize해주는 함수입니다.
                아래와 같은 함수들을 사용할 수 있습니다.
                - lambda x: x.split(' ')
                - Huggingface Tokenizer
                - konlpy.tag의 Mecab

            data_path:
                데이터가 보관되어 있는 경로입니다.

            context_path:
                Passage들이 묶여있는 파일명입니다.

            data_path/context_path가 존재해야합니다.

        Summary:
            Passage 파일을 불러오고 TfidfVectorizer를 선언하는 기능을 합니다.
        """

        with open(context_path, "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로

        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        self.indexer = faiss.read_index(indexer)
        self.q_encoder = torch.load(q_encoder)
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            use_fast=False,
        )
        self.device = "cuda"

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
            breakpoint()
            return result

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

    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--context_path",
        default="../dataset/wikipedia_documents.json",
        type=str,
        help="",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="klue/bert-base",
        type=str,
        help="",
    )
    parser.add_argument(
        "--indexer_path",
        default="./dpr_model/faiss_clusters64_29-23-52.index",
        type=str,
        help="",
    )
    parser.add_argument(
        "--query_encoder",
        default="./dpr_model/q_encoder.pt",
        type=str,
        help="query encoder path",
    )
    args = parser.parse_args()

    # Initiate DenseRetriver
    retriever = DenseRetriever(
        tokenize_fn=args.model_name_or_path,
        indexer=args.indexer_path,
        q_encoder=args.query_encoder,
        context_path=args.context_path,
    )

    # Test single query
    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"
    with timer("single query by faiss"):
        scores, indices = retriever.retrieve_faiss(query)

    # Test bulk dataset query
    # Load dataset and concat train and valid
    org_dataset = load_from_disk("../dataset/train_dataset")
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )
    # query bulk data and caculate accuracy
    with timer("bulk query by exhaustive search"):
        df = retriever.retrieve_faiss(full_ds, topk=3)
        df["correct"] = df["original_context"] == df["context"]

        print("correct retrieval result by faiss", df["correct"].sum() / len(df))
