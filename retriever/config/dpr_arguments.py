from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DprArguments:
    """
    Arguments for training DPR
    """


    output_dir: str = field(
        default="./dpr_model",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )

    model_name_or_path : str = field(
        default='klue/bert-base',
        metadata={
            "help":"model name"
        }
    )
    
    train_dataset: str = field(
        default="../dataset/train_dataset",
        metadata={
            "help":"train dataset"
        }
    )

    learning_rate: float = field(
        default=3e-4,
        metadata={
            "help": "DPR training learning rate"
        },
    )
    per_device_train_batch_size: int = field(
        default=2,
        metadata={
            "help": "DPR training batch size"
        },
    )

    num_train_epochs: int = field(
        default=1,
        metadata={
            "help":"DPR trainin epochs"
        }
    )
    
    num_neg: int = field(
        default=2,
        metadata={
            "help": "Number of in batch negative sample in DPR training"
        }
    )

    weight_decay: float = field(
        default=0.01,
        metadata={
            "help":"weight decay"
        }
    )

    model_save: bool = field(
        default=True,
        metadata={
            "help":"Whether save q-encoder, and p-encoder or not"
        }
    )

    build_faiss: bool = field(
        default=True,
        metadata={
            "help":"Whether build faiss index or not"
        }
    )


    adam_epsilon: float = field(
        default=1e-08
    )

    warmup_steps: float = field(
        default=0
    )

    gradient_accumulation_steps : int = field(
        default=1,
    )

    device: str = field(
        default='cuda'
    )
    