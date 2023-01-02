import json
import os
import pickle
import time
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union

import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm
import unicodedata

from elasticsearch import Elasticsearch
import json
import re
from tqdm import tqdm
import pprint

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

def preprocess(text):
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\\n", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(
        r"[^A-Za-z0-9가-힣.?!,()~‘’“”" ":%&《》〈〉''㈜·\-'+\s一-龥サマーン]", "", text
    )  # サマーン 는 predictions.json에 있었음
    text = re.sub(r"\s+", " ", text).strip()  # 두 개 이상의 연속된 공백을 하나로 치환
    return text


def load_data(dataset_path):

	"""
	dict.fromkeys() 함수 사용으로 모든 document들은 중복이 존재할 시 알아서 제거가 됩니다.
	"""
	with open(dataset_path, "r") as f:
		wiki = json.load(f)

	# wiki document에서 normal document 만 사용할 떄
	# wiki_texts = list(dict.fromkeys([v["text"] for v in wiki.values()]))

	# # wiki document에서 전처리를 진행할 때
	# wiki_texts = list(dict.fromkeys([preprocess(v["text"]) for v in wiki.values()]))

	# # wiki document에서 제목 + 전처리를 진행할 때
	# wiki_texts = list(dict.fromkeys([' '.join([v["title"], preprocess(v["text"])]) for v in wiki.values()]))
	
	# # wiki document에서 제목 + normal document 
	# wiki_texts = list(dict.fromkeys([' '.join([v["title"], v["text"]]) for v in wiki.values()]))

	# # wiki document에서 제목 + 전각문자 처리
	wiki_texts = list(dict.fromkeys([' '.join([v["title"], unicodedata.normalize('NFKC', v["text"])]) for v in wiki.values()]))

	wiki_texts = [text for text in wiki_texts]
	wiki_corpus = [{"document_text":wiki_texts[i]} for i in range(len(wiki_texts))]
	return wiki_corpus

def es_search(es, index_name, question, topk):
    # question = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"
    query = {"query": {"bool": {"must": [{"match": {"document_text": question}}]}}}
    res = es.search(index=index_name, body=query, size=topk)
    return res


class ElasticRetrieval:
	def __init__(
		self,
		data_path: Optional[str] = None, # './dataset',
		context_path: Optional[str] = None, # 'wikipedia_documents.json',
		setting_path: Optional[str] = None, # "./retriever/setting.json",
		index_name: Optional[str] = None, # 'wiki_origin_multi'
	) -> None:

		# Declare ElasticSearch class
		self.es = Elasticsearch("http://localhost:9200", timeout=30, max_retries=10, retry_on_timeout=True)

		# ElasticSearch params
		self.data_path = './dataset' if data_path is None else data_path
		self.context_path = 'wikipedia_documents.json' if context_path is None else context_path
		self.setting_path = './retriever/setting.json' if setting_path is None else setting_path

		if self.es.indices.exists(index=index_name) and (self.es.count(index=index_name)['count'] > 10000):
			# index_name이 현재 존재하고 제대로 데이터가 삽입이 된 상태일때(10000개 이상정도면 이상 없다고 판단)
			self.index_name = index_name
			print(f' ### {index_name} 에서 search를 시작합니다 ###')
			print(f" {index_name} 총 데이터 개수 : {self.es.count(index=self.index_name)['count']}")

		else:
			
			if self.es.indices.exists(index=index_name): self.es.indices.delete(index=index_name)

			with open(self.setting_path, "r") as f:
				setting = json.load(f)
			self.es.indices.create(index=index_name, body=setting)
			self.index_name = index_name
			print("Index creation has been completed")

			wiki_corpus = load_data(os.path.join(self.data_path, self.context_path))
			breakpoint()

			print(f' ### {index_name} 에 데이터를 삽입합니다. ###')
			print(f' 총 데이터 개수 : {len(wiki_corpus)}')

			for i, text in enumerate(tqdm(wiki_corpus)):
				try:
					self.es.index(index=index_name, id=i, body=text)
				except:
					print(f"Unable to load document {i}.")

			n_records = self.es.count(index=index_name)["count"]
			print(f'Successfully loaded {n_records} into {index_name}')
			print("@@@@@@@@@ 데이터 삽입 완료 @@@@@@@@@")

	def retrieve(
		self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
	) -> Union[Tuple[List, List], pd.DataFrame]:

		if isinstance(query_or_dataset, str):
			doc_scoresm, doc_indices, docs = self.get_relevant_doc(
				query_or_dataset, k=topk
			)
			print("[Search query]\n", query_or_dataset, '\n')

			for i in range(min(topk, len(docs))):

				print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
				print(doc_indices[i])
				print(docs[i]["_source"]["document_text"])
			
			return (doc_scores, [doc_indices[i] for i in range(topk)])

		elif isinstance(query_or_dataset, Dataset):

			total = []
			with timer("query exhaustive search"):
				doc_scores, doc_indices, docs = self.get_relevant_doc_bulk(
					query_or_dataset["question"], k=topk
				)

			for idx, example in enumerate(
				tqdm(query_or_dataset, desc="Sparse retrieval with Elasticsearch: ")
			):

				retrieved_context = []
				for i in range(min(topk, len(docs[idx]))):
					retrieved_context.append(docs[idx][i]["_source"]["document_text"])
				
				tmp = {
					"question": example['question'],
					"id": example['id'],
					"context": " ".join(retrieved_context)
				}

				if "context" in example.keys() and "answers" in example.keys():
					tmp["original_context"] = example["context"]
					tmp["answers"] = example["answers"]

				total.append(tmp)
		
		cqas = pd.DataFrame(total)
		return cqas

	def get_relevant_doc(self, query: str, k: Optional[int] =1) -> Tuple[List, List]:
		doc_score = []
		doc_index = []
		res = es_search(self.ex, self.index_name, query, k)
		docs = res["hits"]["hits"]

		for hit in docs:
			doc_score.append(hit["_score"])
			doc_index.append(hit["_id"])
			print("Doc ID: %3r  Score: %5.2f" % (hit["_id"], hit["_score"]))
		
		return doc_score, doc_index, docs
	
	def get_relevant_doc_bulk(
		self, queries: List, k: Optional[int] =1
	) -> Tuple[List, List]:

		total_docs = []
		doc_scores = []
		doc_indices = []

		for query in queries:
			doc_score = []
			doc_index = []
			res = es_search(self.es, self.index_name, query, k)
			docs = res['hits']['hits']

			for hit in docs:
				doc_score.append(hit["_score"])
				doc_indices.append(hit["_id"])

			doc_scores.append(doc_score)
			doc_indices.append(doc_index)
			total_docs.append(docs)

		return doc_scores, doc_indices, total_docs