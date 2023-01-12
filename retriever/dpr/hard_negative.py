import json
import os
import pickle
import time
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union

import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk, load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm
import unicodedata

from elasticsearch import Elasticsearch
import json
import re
from tqdm import tqdm
import pprint


def es_search(es, index_name, question, topk):
    query = {"query": {"bool": {"must": [{"match": {"document_text": question}}]}}}
    res = es.search(index=index_name, body=query, size=topk)
    return res


class HardNegativeRetrieval:
    def __init__(
        self,
        setting_path: Optional[str] = None,  # "./config/setting.json",
        index_name: Optional[str] = None,  # basic_dataset, squad, ai_hub
    ) -> None:

        # Declare ElasticSearch class
        self.es = Elasticsearch(
            "http://localhost:9200", timeout=30, max_retries=10, retry_on_timeout=True
        )
        self.setting_path = (
            "./config/setting.json" if setting_path is None else setting_path
        )
        self.index_name = index_name

    def check_index(self):

        index_name = self.index_name
        if self.es.indices.exists(index=index_name):
            print(f" @#@#@# {index_name} is aleray exist in elastic search @#@#@# ")
            return True
        else:
            print(f"{index_name} is not aleray exist in elastic search")
            print(f" Creating '{index_name}' in elastic search ")
            self._create_index()

    def _create_index(self):

        index_name = self.index_name

        assert index_name in [
            "basic_dataset",
            "squad",
            "ai_hub",
        ], "hard_negative용 index는 \['basic_dataset','squad','aihub'] 중에 하나여야 합니다."

        try:
            self.es.indices.delete(index=index_name)
        except:
            pass

        with open(self.setting_path, "r") as f:
            setting = json.load(f)

        self.es.indices.create(index=index_name, body=setting)
        self.index_name = index_name
        print("Index creation has been completed")

        if index_name == "basic_dataset":

            dataset = load_from_disk("../../dataset/train_dataset")
            dataset = concatenate_datasets(
                [
                    dataset["train"].flatten_indices(),
                    dataset["validation"].flatten_indices(),
                ]
            )

            dataset_corpus = list(dict.fromkeys([each for each in dataset["context"]]))

        elif index_name == "squad":

            dataset = load_dataset("sangmun2/squad_train")
            dataset_corpus = list(
                dict.fromkeys([each for each in dataset["train"]["context"]])
            )

        elif index_name == "ai_hub":

            dataset = load_dataset("sangmun2/ai_hub_qa_without_dup")
            dataset_corpus = list(
                dict.fromkeys([each for each in dataset["train"]["context"]])
            )

        dataset_corpus = [
            {"document_text": dataset_corpus[i]} for i in range(len(dataset_corpus))
        ]

        print(f" ### {index_name} 에 데이터를 삽입합니다. ###")
        print(f" 총 데이터 개수 : {len(dataset_corpus)}")

        for i, text in enumerate(tqdm(dataset_corpus)):
            try:
                self.es.index(index=index_name, id=i, body=text)
            except:
                print(f"Unable to load document {i}.")

        n_records = self.es.count(index=index_name)["count"]
        print(f"Successfully loaded {n_records} into {index_name}")
        print("@@@@@@@@@ 데이터 삽입 완료 @@@@@@@@@")

    def retrieve_HN(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices, docs = self.get_relevant_doc(
                query_or_dataset, k=topk
            )

            return (doc_scores, [each["_source"]["document_text"] for each in docs])

    def delete_index(self):

        index_name = self.index_name
        self.es.indices.delete(index=index_name)
