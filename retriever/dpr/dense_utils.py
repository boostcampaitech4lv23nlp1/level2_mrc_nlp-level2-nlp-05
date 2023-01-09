from typing import Callable, Dict, List, NoReturn, Optional, Tuple

from datasets import load_from_disk, load_dataset
from torch.utils.data import Dataset, TensorDataset
import numpy as np
from tqdm import tqdm
from elastic_retrieval import ElasticRetrieval


class BaseDataset(Dataset):
    def __init__(
        self,
        tokenizer: Callable[[str], List[str]],
        datapath: str,
        max_question_seq_length: Optional[int] = None,
        max_context_seq_length: Optional[int] = None,
        in_batch_neg: Optional[bool] = False,
        num_neg: Optional[int] = None,
        hard_neg: Optional[bool] = False,
    ):

        load_datasets = load_from_disk(dataset_path=datapath)

        if in_batch_neg:

            print(" ### making in_batch_neg ### ")
            corpus = np.array(
                list(set([example for example in load_datasets["context"]]))
            )
            p_with_neg = []

            for context in tqdm(load_datasets["context"], desc="iter in batch neg"):

                while True:
                    neg_idx = np.random.randint(len(corpus), size=num_neg)

                    if not context in corpus[neg_idx]:
                        p_neg = corpus[neg_idx]
                        p_with_neg.append(context)
                        p_with_neg.extend(p_neg)
                        break

        if hard_neg:

            print(" ### making in_batch_neg + Hard negative(elastic) ### ")
            retriever = ElasticRetrieval(index_name="basic_dataset")
            retriever.check_index()

            print
            corpus = np.array(
                list(set([example for example in load_datasets["context"]]))
            )
            p_with_neg = []

            for context, query in tqdm(
                zip(load_datasets["context"], load_datasets["question"]),
                total=len(load_datasets["context"]),
                desc="iter in batch neg + hard_neg",
            ):

                while True:
                    neg_idx = np.random.randint(len(corpus), size=num_neg)
                    scores, docs = retriever.retrieve_HN(query, topk=2)

                    if not context in corpus[neg_idx]:
                        tmp_doc = docs[1] if context == docs[0] else docs[0]
                        p_neg = corpus[neg_idx]
                        p_with_neg.append(context)
                        p_with_neg.extend(p_neg)
                        p_with_neg.append(tmp_doc)
                        break

        # query와 passage를 위한 sequence를 tokenization
        q_seqs = tokenizer(
            load_datasets["question"],
            # max_length=max_question_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        p_seqs = tokenizer(
            p_with_neg if in_batch_neg or hard_neg else load_datasets["context"],
            # max_length=max_context_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        if in_batch_neg:
            max_len = p_seqs["input_ids"].size(-1)
            p_seqs["input_ids"] = p_seqs["input_ids"].view(-1, num_neg + 1, max_len)
            p_seqs["attention_mask"] = p_seqs["attention_mask"].view(
                -1, num_neg + 1, max_len
            )
            p_seqs["token_type_ids"] = p_seqs["token_type_ids"].view(
                -1, num_neg + 1, max_len
            )

        if hard_neg:
            max_len = p_seqs["input_ids"].size(-1)
            p_seqs["input_ids"] = p_seqs["input_ids"].view(-1, num_neg + 2, max_len)
            p_seqs["attention_mask"] = p_seqs["attention_mask"].view(
                -1, num_neg + 2, max_len
            )
            p_seqs["token_type_ids"] = p_seqs["token_type_ids"].view(
                -1, num_neg + 2, max_len
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
    def __init__(
        self,
        tokenizer: Callable[[str], List[str]],
        datapath: str,
        max_question_seq_length: Optional[int] = None,
        max_context_seq_length: Optional[int] = None,
        in_batch_neg: Optional[bool] = False,
        num_neg: Optional[int] = None,
        hard_neg: Optional[bool] = False,
    ):

        load_datasets = load_dataset("sangmun2/squad_train")

        if in_batch_neg:

            print(" ### making in_batch_neg ### ")
            corpus = np.array(
                list(set([example for example in load_datasets["context"]]))
            )
            p_with_neg = []

            for context in tqdm(load_datasets["context"], desc="iter in batch neg"):

                while True:
                    neg_idx = np.random.randint(len(corpus), size=num_neg)

                    if not context in corpus[neg_idx]:
                        p_neg = corpus[neg_idx]
                        p_with_neg.append(context)
                        p_with_neg.extend(p_neg)
                        break

        if hard_neg:

            print(" ### making in_batch_neg + Hard negative(elastic) ### ")
            retriever = ElasticRetrieval(index_name="squad")
            retriever.check_index()

            corpus = np.array(
                list(set([example for example in load_datasets["context"]]))
            )
            p_with_neg = []

            for context, query in tqdm(
                zip(load_datasets["context"], load_datasets["question"]),
                total=len(load_datasets["context"]),
                desc="iter in batch neg + hard_neg",
            ):

                while True:
                    neg_idx = np.random.randint(len(corpus), size=num_neg)
                    scores, docs = retriever.retrieve_HN(query, topk=2)

                    if not context in corpus[neg_idx]:
                        tmp_doc = docs[1] if context == docs[0] else docs[0]
                        p_neg = corpus[neg_idx]
                        p_with_neg.append(context)
                        p_with_neg.extend(p_neg)
                        p_with_neg.append(tmp_doc)
                        break

        # query와 passage를 위한 sequence를 tokenization
        q_seqs = tokenizer(
            load_datasets["question"],
            # max_length=max_question_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        p_seqs = tokenizer(
            p_with_neg if in_batch_neg or hard_neg else load_datasets["context"],
            # max_length=max_context_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        if in_batch_neg:
            max_len = p_seqs["input_ids"].size(-1)
            p_seqs["input_ids"] = p_seqs["input_ids"].view(-1, num_neg + 1, max_len)
            p_seqs["attention_mask"] = p_seqs["attention_mask"].view(
                -1, num_neg + 1, max_len
            )
            p_seqs["token_type_ids"] = p_seqs["token_type_ids"].view(
                -1, num_neg + 1, max_len
            )

        if hard_neg:
            max_len = p_seqs["input_ids"].size(-1)
            p_seqs["input_ids"] = p_seqs["input_ids"].view(-1, num_neg + 2, max_len)
            p_seqs["attention_mask"] = p_seqs["attention_mask"].view(
                -1, num_neg + 2, max_len
            )
            p_seqs["token_type_ids"] = p_seqs["token_type_ids"].view(
                -1, num_neg + 2, max_len
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


class BothDataset(Dataset):
    def __init__(
        self,
        tokenizer: Callable[[str], List[str]],
        datapath: str,
        max_question_seq_length: Optional[int] = None,
        max_context_seq_length: Optional[int] = None,
        in_batch_neg: Optional[bool] = False,
        num_neg: Optional[int] = None,
        hard_neg: Optional[bool] = False,
    ):

        # load_datasets = load_dataset('sangmun2/squad_train')

        load_datasets = load_from_disk(dataset_path=datapath)

        korquad_dataset = load_dataset("sangmun2/squad_train", use_auth_token=True)
        korquad_dataset = korquad_dataset["train"][:3000]

        querys = load_datasets["question"] + korquad_dataset["question"]
        contexts = load_datasets["context"] + korquad_dataset["context"]

        if in_batch_neg:

            print(" ### making in_batch_neg ### ")
            # corpus = np.array(list(set([example for example in load_datasets['context']])))
            corpus = np.array(list(set([example for example in contexts])))
            p_with_neg = []

            # for context in tqdm(load_datasets['context'], desc="iter in batch neg"):
            for context in tqdm(contexts, desc="iter in batch neg"):

                while True:
                    neg_idx = np.random.randint(len(corpus), size=num_neg)

                    if not context in corpus[neg_idx]:
                        p_neg = corpus[neg_idx]
                        p_with_neg.append(context)
                        p_with_neg.extend(p_neg)
                        break

        if hard_neg:

            print(" ### making in_batch_neg + Hard negative(elastic) ### ")
            retriever = ElasticRetrieval(index_name="squad")
            retriever.check_index()

            # corpus = np.array(list(set([example for example in load_datasets['context']])))
            corpus = np.array(list(set([example for example in contexts])))
            p_with_neg = []

            # for context, query in tqdm(zip(load_datasets['context'], load_datasets['question']),
            for context, query in tqdm(
                zip(contexts, querys),
                total=len(contexts),
                desc="iter in batch neg + hard_neg",
            ):

                while True:
                    neg_idx = np.random.randint(len(corpus), size=num_neg)
                    scores, docs = retriever.retrieve_HN(query, topk=2)

                    if not context in corpus[neg_idx]:
                        tmp_doc = docs[1] if context == docs[0] else docs[0]
                        p_neg = corpus[neg_idx]
                        p_with_neg.append(context)
                        p_with_neg.extend(p_neg)
                        p_with_neg.append(tmp_doc)
                        break

        # query와 passage를 위한 sequence를 tokenization
        q_seqs = tokenizer(
            querys,
            # max_length=max_question_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        p_seqs = tokenizer(
            p_with_neg if in_batch_neg or hard_neg else contexts,
            # max_length=max_context_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        if in_batch_neg:
            max_len = p_seqs["input_ids"].size(-1)
            p_seqs["input_ids"] = p_seqs["input_ids"].view(-1, num_neg + 1, max_len)
            p_seqs["attention_mask"] = p_seqs["attention_mask"].view(
                -1, num_neg + 1, max_len
            )
            p_seqs["token_type_ids"] = p_seqs["token_type_ids"].view(
                -1, num_neg + 1, max_len
            )

        if hard_neg:
            max_len = p_seqs["input_ids"].size(-1)
            p_seqs["input_ids"] = p_seqs["input_ids"].view(-1, num_neg + 2, max_len)
            p_seqs["attention_mask"] = p_seqs["attention_mask"].view(
                -1, num_neg + 2, max_len
            )
            p_seqs["token_type_ids"] = p_seqs["token_type_ids"].view(
                -1, num_neg + 2, max_len
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
