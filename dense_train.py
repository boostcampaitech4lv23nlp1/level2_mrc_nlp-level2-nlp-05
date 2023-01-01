import argparse
import os
import random

import numpy as np
import torch
from datasets import load_from_disk
from transformers import (AutoTokenizer, BertModel, BertPreTrainedModel,
                          TrainingArguments)

from utils.utils_dr import BaseDataset
from trainer.trainer_dr import DenseRetrievalTrainer


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
    training_args = cfg.training_args
    model_args = cfg.model_args

    model_name = model_args.dense_train_model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_path = data_args.train_dataset_name
    valid_path = data_args.valid_dataset_name

    train_dataset = BaseDataset(tokenizer=tokenizer, datapath=train_path)
    valid_dataset = load_from_disk(dataset_path=valid_path)

    print("Train Dataset Length:", len(train_dataset))
    print("Valid Dataset Length:", len(valid_dataset))

    p_encoder = BertEncoder.from_pretrained(model_name)
    q_encoder = BertEncoder.from_pretrained(model_name)

    # yaml
    args = TrainingArguments(
        output_dir="./dense_retireval",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=20,
        weight_decay=0.01,
        gradient_accumulation_steps=8,
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


if __name__ == "__main__":
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    args, _ = parser.parse_known_args()
    cfg = OmegaConf.load(f"./config/{args.config}/dense_config.yaml")

    main(cfg)
