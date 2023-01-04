import argparse
import os
import random

import numpy as np
import torch
from datasets import load_from_disk
from transformers import (AutoTokenizer, BertModel, BertPreTrainedModel,
                          TrainingArguments)

from dense_utils import BaseDataset, KorquadDataset, BothDataset
from dense_trainer import DenseRetrievalTrainer
from dense_model import BertEncoder

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


def main(cfg):
    seed_everything(42)
    data_path = "./dataset/"
    context_path = "wikipedia_documents.json"
    data_args = cfg.data_args
    train_args = cfg.training_args
    model_args = cfg.model_args

    model_name = model_args.dense_train_model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_path = data_args.train_dataset_name
    valid_path = data_args.valid_dataset_name

    if data_args.dataset == 'basic':
        train_dataset = BaseDataset(tokenizer=tokenizer,
                                    datapath=train_path,
                                    in_batch_neg=train_args.in_batch_neg,
                                    num_neg=train_args.num_neg,
                                    hard_neg=train_args.hard_neg) 

    if data_args.dataset == 'squad':
        train_dataset = KorquadDataset(tokenizer=tokenizer,
                                    datapath=train_path,
                                    in_batch_neg=train_args.in_batch_neg,
                                    num_neg=train_args.num_neg,
                                    hard_neg=train_args.hard_neg) # KorquadDataset 추가

    if data_args.dataset == 'both':
        train_dataset = BothDataset(tokenizer=tokenizer,
                                    datapath=train_path,
                                    in_batch_neg=train_args.in_batch_neg,
                                    num_neg=train_args.num_neg,
                                    hard_neg=train_args.hard_neg) # BothDataset 추가


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
        warmup_ratio=train_args.warmup_ratio
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

    DR_trainer.train()
    DR_trainer.build_faiss(p_encoder_path='./retriever/saved_models/p_encoder', num_clusters=32)


if __name__ == "__main__":
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    args, _ = parser.parse_known_args()
    cfg = OmegaConf.load(f"./retriever/config/{args.config}dense_config.yaml")

    main(cfg)
