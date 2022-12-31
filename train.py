import logging
import os
import sys
from datetime import datetime
from typing import NoReturn
import argparse
import pandas as pd

from arguments import DataTrainingArguments, ModelArguments
from datasets import DatasetDict, load_from_disk, load_metric, Features, Value, Sequence
from datasets import Dataset, concatenate_datasets, load_from_disk, load_dataset
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
from data_loaders.data_loader import load_train_dataset, load_eval_dataset
from run_mrc import run_mrc
from run_sparse_retrieval import run_sparse_retrieval
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def main(model_args, data_args, training_args):

    # Redefine training arguments
    training_args = TrainingArguments(**training_args)

    now = datetime.now()
    train_start_time = now.strftime("%d-%H-%M")
    training_args.output_dir = os.path.join(
        training_args.output_dir, f"{train_start_time}"
    )

    # Set Loggin & verbosity
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set Seed
    set_seed(training_args.seed)

    # Dataset Load
    datasets = load_from_disk(data_args.dataset_name)

    # AutoConfig, Autotokenizer, AutoModel
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name is not None
        else model_args.model_name_or_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name is not None
        else model_args.model_name_or_path,
        use_fast=True,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )

    # print training configuration
    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")
    print(datasets)

    print("train:", training_args.do_train)
    print("eval:", training_args.do_eval)
    print("predict:", training_args.do_predict)
    # do_train mrc model 혹은 do_eval mrc model
    if training_args.do_train or training_args.do_eval or training_args.do_predict:
        print("run_mrc...")
        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    args, _ = parser.parse_known_args()
    cfg = OmegaConf.load(f"./config/{args.config}.yaml")

    main(cfg.model_args, cfg.data_args, cfg.training_args)
