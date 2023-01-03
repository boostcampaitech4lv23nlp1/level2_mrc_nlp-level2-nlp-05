from typing import Callable, Dict, List, NoReturn, Optional, Tuple

from datasets import load_from_disk, load_dataset
from torch.utils.data import Dataset, TensorDataset

class BaseDataset(Dataset):
    def __init__(
        self,
        tokenizer: Callable[[str], List[str]],
        datapath: str,
        max_question_seq_length: Optional[int] = None,
        max_context_seq_length: Optional[int] = None,
    ):

        load_datasets = load_from_disk(dataset_path=datapath)

        # query와 passage를 위한 sequence를 tokenization
        q_seqs = tokenizer(
            load_datasets["question"],
            # max_length=max_question_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        p_seqs = tokenizer(
            load_datasets["context"],
            # max_length=max_context_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # 데이터셋을 학습하기 위해서 TensorDataset으로 변경
        self.dataset = TensorDataset(
            p_seqs["input_ids"],
            p_seqs["attention_mask"],
            p_seqs["token_type_ids"],
            q_seqs["input_ids"],
            q_seqs["attention_mask"],
            q_seqs["token_type_ids"],
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class KorquadDataset(Dataset):
    """ KorQuAD Dataset 추가 학습

    Args:
        Dataset (_type_): _description_
    """
    def __init__(
        self,
        tokenizer: Callable[[str], List[str]],
        datapath: str,
        max_question_seq_length: Optional[int] = None,
        max_context_seq_length: Optional[int] = None,
    ):

        load_datasets = load_from_disk(dataset_path=datapath)
        
        korquad_dataset = load_dataset("sangmun2/squad_train", use_auth_token=True)
        korquad_dataset = korquad_dataset['train'][:5000]
        
        querys = load_datasets["question"] + korquad_dataset["question"]
        passages = load_datasets["context"] + korquad_dataset["context"]
        
        # query와 passage를 위한 sequence를 tokenization
        q_seqs = tokenizer(
            querys,
            # max_length=max_question_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        p_seqs = tokenizer(
            passages,
            # max_length=max_context_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # 데이터셋을 학습하기 위해서 TensorDataset으로 변경
        self.dataset = TensorDataset(
            p_seqs["input_ids"],
            p_seqs["attention_mask"],
            p_seqs["token_type_ids"],
            q_seqs["input_ids"],
            q_seqs["attention_mask"],
            q_seqs["token_type_ids"],
        )
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


# TODO : Hard negative
class HardNegativeRandomDataset(Dataset):
    """_summary_

    Args:
        Dataset (_type_): _description_
    """
    def __init__(
        self,
        tokenizer: Callable[[str], List[str]],
        datapath: str,
        max_question_seq_length: Optional[int] = None,
        max_context_seq_length: Optional[int] = None,
    ):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
