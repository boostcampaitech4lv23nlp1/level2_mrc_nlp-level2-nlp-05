import logging
import sys
from typing import Callable, Dict, List, NoReturn, Tuple

import numpy as np
from arguments import DataTrainingArguments, ModelArguments
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
    load_from_disk,
    load_metric,
)
from retriever.sparse_retrieval import SparseRetrieval
from retriever.faiss_retrieval import FaissRetrieval
from retriever.elastic_retrieval import ElasticRetrieval
from retriever.dense_retrieval import DenseRetrieval
from trainer.trainer_qa import QuestionAnsweringTrainer
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from utils.utils_qa import check_no_error, postprocess_qa_predictions
import pandas as pd


def run_retrieval(
    tokenize_fn: Callable[[str], List[str]],
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    data_path: str = "./dataset",
    context_path: str = "wikipedia_documents.json",
) -> DatasetDict:

    # Query에 맞는 Passage들을 Retrieval 합니다.

    if data_args.retriever_type == "faiss":
        print(data_args.retriever_type)
        retriever = FaissRetrieval(
            tokenize_fn=tokenize_fn, data_path=data_path, context_path=context_path
        )
        retriever.get_sparse_embedding()
        retriever.build_faiss(num_clusters=data_args.num_clusters)
        df = retriever.retrieve_faiss(
            datasets["validation"], topk=data_args.top_k_retrieval
        )

    elif data_args.retriever_type == "elastic":
        print(data_args.retriever_type)

        retriever = ElasticRetrieval(
            data_path=data_args.data_path,
            context_path=data_args.context_path,
            setting_path=data_args.setting_path,
            index_name=data_args.index_name,
        )

        df = retriever.retrieve(datasets["validation"], topk=data_args.top_k_retrieval)

    elif data_args.retriever_type == "base":
        print(data_args.retriever_type)
        retriever = SparseRetrieval(
            tokenize_fn=tokenize_fn, data_path=data_path, context_path=context_path
        )
        retriever.get_sparse_embedding()
        df = retriever.retrieve(datasets["validation"], topk=data_args.top_k_retrieval)

    elif data_args.retriever_type == "ensemble":
        df = pd.read_csv(data_args.csv_ensemble_path)

    # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
    f = Features(
        {
            "context": Value(dtype="string", id=None),
            "id": Value(dtype="string", id=None),
            "question": Value(dtype="string", id=None),
        }
    )
    predict_datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})

    return predict_datasets
