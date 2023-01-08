import json
import os
import pickle
import time
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union

import faiss
import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk, load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm
from sparse_retrieval import SparseRetrieval
from faiss_retrieval import FaissRetrieval
from elastic_retrieval import ElasticRetrieval


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--dataset_name", default="../dataset/train_dataset", type=str, help=""
    )
    parser.add_argument(
        "--model_name_or_path",
        default="bert-base-multilingual-cased",
        type=str,
        help="",
    )
    parser.add_argument("--data_path", default="../dataset", type=str, help="")
    parser.add_argument(
        "--context_path", default="wikipedia_documents.json", type=str, help=""
    )
    parser.add_argument("--retriever_type", default="elastic", type=str, help="")
    args = parser.parse_args()

    print(args)
    # Test sparse
    org_dataset = load_from_disk(args.dataset_name)
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
    )

    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

    print(args.retriever_type)
    if args.retriever_type == "faiss":
        retriever = FaissRetrieval(
            tokenize_fn=tokenizer.tokenize,
            data_path=args.data_path,
            context_path=args.context_path,
        )
        retriever.get_sparse_embedding()
        retriever.build_faiss(num_clusters=64)
        # test single query
        with timer("single query by faiss"):
            scores, indices = retriever.retrieve_faiss(query)

        # test bulk
        with timer("bulk query by faiss search"):
            df = retriever.retrieve_faiss(full_ds)
            df["correct"] = df["original_context"] == df["context"]

            print("correct retrieval result by faiss", df["correct"].sum() / len(df))
            # 0.03482824427480916 점

    elif args.retriever_type == "elastic":
        print("init elastic...")
        retriever = ElasticRetrieval(index_name="squad")
        retriever.check_index()
        retriever.create_index()
        
        # dataset = load_dataset('sangmun2/ai_hub_qa_without_dup')

        breakpoint()

        with timer("single query by elastic search"):
            scores, docs = retriever.retrieve_HN(query, topk=2)

        # with timer("bulk query by elastic search"):
        #     df = retriever.retrieve(full_ds)
        #     df["correct"] = df["original_context"] == df["context"]
        #     print(
        #         "correct retrieval result by elastic search",
        #         df["correct"].sum() / len(df),
        #     )

        #     df.to_csv("elastic.csv")
        #     # 0.57490458015267186 점

    else:
        retriever = SparseRetrieval(
            tokenize_fn=tokenizer.tokenize,
            data_path=args.data_path,
            context_path=args.context_path,
        )
        retriever.get_sparse_embedding()

        with timer("single query by exhaustive search"):
            scores, indices = retriever.retrieve(query)

        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve(full_ds)
            df["correct"] = df["original_context"] == df["context"]
            print(
                "correct retrieval result by exhaustive search",
                df["correct"].sum() / len(df),
            )  # 0.25190839694656486 점
