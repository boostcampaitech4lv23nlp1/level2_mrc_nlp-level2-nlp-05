## DPR Manal

### 1. Dataset
dense_utils.py에서 확인 가능

* BaseDataset 사용 시 → `dataset: basic` 으로 설정
* KorquadDataset 사용 시 → `dataset: both` 으로 설정

### 2. Negative train

* in_batch_neg 사용 시 →   `in_batch_neg : True, hard_neg : False` 으로 설정
* hard_neg 사용 시 -> `in_batch_neg : False, hard_neg : True` 으로 설정


### 3. How DPR model train
```python dense_train.py```