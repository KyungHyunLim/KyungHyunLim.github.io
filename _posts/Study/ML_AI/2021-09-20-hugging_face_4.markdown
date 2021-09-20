---
layout: post
title:  "Huggingface- Chapter 4. Sharing models and tokenizers"
date:   2021-09-20 17:45:22
categories: [NLP, ML_AI]
use_math: true
---

# Chapter 4. Sharing models and tokenizers
## 1. Task
* Hugging face에서 정의된 task tag 사용
    * 다른 task의 pretrained 모델을 불러오면, head의 구조가 다르기 때문에 적절한 결과를 얻기 어려움

## 2. Load
* 1과 같이 직접 부를 수도 있지만, 2처럼 Auto*class를 사용하는 것을 추천
* 1번과 같은 코드는 Camembert에 한정될 수 밖에 없기 때문

```python
# 1.
from transformers import CamembertTokenizer, CamembertForMaskedLM 

tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
model = CamembertForMaskedLM.from_pretrained("camembert-base")

# 2.
from transformers import AutoTokenizer, AutoModelForMaskedLM 

tokenizer = AutoTokenizer.from_pretrained("camembert-base")
model = AutoModelForMaskedLM.from_pretrained("camembert-base")
```

## 3. Sharing
* Hugging Face Hub를 통해 간단하게 sharing 가능
* 작성중...