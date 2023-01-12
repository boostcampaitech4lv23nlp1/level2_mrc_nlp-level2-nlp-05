## Salient Span Masking

- **Salient Span Masking(SSM)** 은 REALM(https://arxiv.org/pdf/2002.08909.pdf)에서 제시된 pretraining방법으로 인물, 날짜, 장소, 수량과 같을 Named Entity를 Masking하여 모델이 QA task에 적합한 world knowledge를 더 잘 학습할 수 있도록 한다. 여기에는 Reader 모델이 읽고 이해해야 하는 wikipedea문서에서 Named Entity를 마스킹하는 코드와, 마스킹된 데이터셋을 이용해 MLM학습을 수행하는 코드가 포함된다.

---
### 디렉토리 설명
- ssm_data_wiki.ipynb : wikipedia dataset에서 Named Entity를 마스킹해 저장
- ssm_data_qa.ipynb : qa dataset에서 정답 token을 마스킹해 저장
- mlm.py : 15%의 token을 무작위로 마스킹해 mlm
- ssm.py : Named Entity나 정답 token을 미리 마스킹해놓은 데이터를 이용해 mlm

```python
├──📁ssm 
│   ├── ssm_data_wiki.ipynb  
│   ├── ssm_data_qa.ipynb 
│   ├── mlm.py 
│   ├── ssm.py
│   └──📁pretrained # pretrained 모델 저장 경로 
```

---
### Load Pretrained Model
- model = AutoModel.from_pretrained('./ssm/pretrained') 으로 불러옴
