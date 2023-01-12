from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="klue/roberta-large",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="./dataset/train_dataset",
        metadata={"help": "The name of the dataset to use."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        },
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    eval_retrieval: bool = field(
        default=False,
        metadata={"help": "Whether to run passage retrieval using sparse embedding."},
    )
    num_clusters: int = field(
        default=64, metadata={"help": "Define how many clusters to use for faiss."}
    )
    top_k_retrieval: int = field(
        default=10,
        metadata={
            "help": "Define how many top-k passages to retrieve based on similarity."
        },
    )
    use_faiss: bool = field(
        default=False, metadata={"help": "Whether to build with faiss"}
    )


@dataclass
class TrainingArguments(TrainingArguments):

    now = datetime.now()
    train_start_time = now.strftime("%d-%H-%M")
    output_dir: str = field(
        default=f"./models/{train_start_time}", metadata={"help": "Saved result path"}
    )

    logging_dir: str = field(
        default="./logs", metadata={"help": "Logging directory path"}
    )

    per_device_train_batch_size: int = field(
        default=16, metadata={"help": "Train batch size per device"}
    )

    per_device_eval_batch_size: int = field(
        default=16, metadata={"help": "Eval batch size per device"}
    )

    save_total_limit: int = field(
        default=1, metadata={"help": "limitation of number of saved file"}
    )

    load_best_model_at_end: bool = field(
        default=False, metadata={"help": "Load best model at end"}
    )

    num_train_epochs: int = field(default=2, metadata={"help": "Train epochs"})

    evaluation_strategy: str = field(
        default="steps", metadata={"help": "evaluation strategy"}
    )

    logging_steps: int = field(default=20, metadata={"help": "logging steps"})

    eval_steps: int = field(
        default=200, metadata={"help":"evaludation steps"}
    )
    
    overwrite_output_dir: bool = field(
        default=True,
        metadata={"help": "Whether overwrite output_dir when output_dir already exist"},
    )

    do_train: bool = field(default=True, metadata={"help": "whether train or not"})

    do_eval: bool = field(default=False, metadata={"help": "whether eval or not"})

    do_predict: bool = field(default=False, metadata={"help": "whether predict or not"})

    warmup_ratio: float = field(default=0.1, metadata={"help": "Train warmup ratio"})

    learning_rate: float = field(default=5e-5, metadata={"help": "train learning rate"})
