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

from elasticsearch import Elasticsearch
import json
import re
from tqdm import tqdm
import pprint


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


def load_data(dataset_path):
    # dataset_path = "../data/wikipedia_documents.json"
    with open(dataset_path, "r") as f:
        wiki = json.load(f)

    wiki_texts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
    wiki_texts = [text for text in wiki_texts]
    wiki_corpus = [{"document_text": wiki_texts[i]} for i in range(len(wiki_texts))]
    return wiki_corpus


def preprocess(text):
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\\n", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(
        r"[^A-Za-z0-9가-힣.?!,()~‘’“”" ":%&《》〈〉''㈜·\-'+\s一-龥サマーン]", "", text
    )  # サマーン 는 predictions.json에 있었음
    text = re.sub(r"\s+", " ", text).strip()  # 두 개 이상의 연속된 공백을 하나로 치환

    return text


def es_search(es, index_name, question, topk):
    # question = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"
    query = {"query": {"bool": {"must": [{"match": {"document_text": question}}]}}}
    res = es.search(index=index_name, body=query, size=topk)
    return res


class ElasticRetrieval:
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "../dataset/",
        context_path: Optional[str] = "wikipedia_documents.json",
        setting_path: Optional[str] = "./retriever/setting.json",
        index_name: Optional[str] = "origin-wiki-multi",
    ) -> None:

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

        """

        self.es = Elasticsearch(
            "http://localhost:9200", timeout=30, max_retries=10, retry_on_timeout=True
        )

        print(index_name)
        # self.es.indices.delete(index=index_name)
        if self.es.indices.exists(index=index_name):
            # origin-wiki가 이미 존재한다면
            self.index_name = index_name
        else:
            # origin-wiki가 존재하지 않는다면
            # index 생성
            with open(setting_path, "r") as f:
                setting = json.load(f)
            self.es.indices.create(index=index_name, body=setting)
            self.index_name = index_name
            print("Index creation has been completed")

            # wiki_corpus 로드해서 text preprocessing 후 삽입
            wiki_corpus = load_data(os.path.join(data_path, context_path))
            for i, text in enumerate(tqdm(wiki_corpus)):
                try:
                    self.es.index(index=index_name, id=i, body=text)
                except:
                    print(f"Unable to load document {i}.")

            n_records = self.es.count(index=index_name)["count"]
            print(f"Succesfully loaded {n_records} into {index_name}")
            print("@@@@@@@ 데이터 삽입 완료 @@@@@@@")

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices, docs = self.get_relevant_doc(
                query_or_dataset, k=topk
            )
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(min(topk, len(docs))):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(doc_indices[i])
                print(docs[i]["_source"]["document_text"])

            return (doc_scores, [doc_indices[i] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):
            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices, docs = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )

            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval with Elasticsearch: ")
            ):
                # retrieved_context 구하는 부분 수정
                retrieved_context = []
                for i in range(min(topk, len(docs[idx]))):
                    retrieved_context.append(docs[idx][i]["_source"]["document_text"])

                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    # "context_id": doc_indices[idx],
                    "context": " ".join(retrieved_context),  # 수정
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        doc_score = []
        doc_index = []
        res = es_search(self.es, self.index_name, query, k)
        docs = res["hits"]["hits"]

        for hit in docs:
            doc_score.append(hit["_score"])
            doc_index.append(hit["_id"])
            print("Doc ID: %3r  Score: %5.2f" % (hit["_id"], hit["_score"]))

        return doc_score, doc_index, docs

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:
        total_docs = []
        doc_scores = []
        doc_indices = []

        for query in queries:
            doc_score = []
            doc_index = []
            res = es_search(self.es, self.index_name, query, k)
            docs = res["hits"]["hits"]

            for hit in docs:
                doc_score.append(hit["_score"])
                doc_indices.append(hit["_id"])

            doc_scores.append(doc_score)
            doc_indices.append(doc_index)
            total_docs.append(docs)

        return doc_scores, doc_indices, total_docs
