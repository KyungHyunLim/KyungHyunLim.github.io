---
layout: post
title:  "Huggingface- Text Classification"
date:   2021-09-22 23:45:22
categories: [NLP, ML_AI]
use_math: true
---

# 1. GLUE
## 1.1 GLUE tasks
* CoLA (Corpus of Linguistic Acceptability)
    * sentence가 문법적으로 맞는지 틀린지 판별
* MNLI (Multi-Genre Natural Language Inference)
    * 주어진 hypothesis과 관련이 있는지, 모순이 있는지, 관련이 없는지 추론
* MRPC (Microsoft Research Paraphrase Corpus)
    * 두 sentence가 다른 하나로 부터 paraphrase된 것인지 아닌지 판단
* QNLI (Question-answering Natural Language Inference)
    * 한 질문에 대한 정답이 두번째 sentence에 있는지 판별
* QQP (Quora Question Pairs2)
    * 두 질문이 의미적으로 동일한지 아닌지 판별
* RTE (Recognizing Textual Entailment)
    * 한 문장이 주거진 가정을 내포하는지 아닌지 판별
* SST-2 (Stanford Sentiment Treebank)
    * 문장이 긍정적인지 부정적인지 판별
* STS-B (Semantic Textual Similarity Benchmark)
    * 두 문장의 유사도를 1~5 score로 결정
* WNLI (Winograd Natural Language Inference)
    * anonymous pronoun이 있는 문장과 이 pronoun이 대체된 문장이 수반되는지 여부 확인

```python
# cola로 설정하면 text classification task load 가능
actual_task = "cola"
dataset = load_dataset("glue", actual_task)
metric = load_metric('glue', actual_task)
```

## 1.2 Metrics
* Accuracy: 정확도
* F1 score: precision과 recall의 조화평균
* Pearson Correlation
    * 두 변수 X와 Y간의 선형 상관 관계
    * -1 ~ 1 사이 값
        * 클수록 상관관계가 높음
        * 즉, 1은 완전한 선형관계를 의미
    * ${공분산 \over 표준편차 \cdot 표준편차}$
* Spearman Correlation
    * 두 변수의 순위 사이의 통계적 의존성을 측정
    * 단순히 한 변수가 증가할 때, 다른 변수가 증가하는지 감소하는지에 대한 관계만들 나타냄
    * -1 ~ 1 사이 값
    * E.g.  
        ![](/assets/image/ustagelv2/hf_1.jpg)
        * 위와 같은 경우, x가 증가할때 y도 증가하지만 그 양이 일정하지 않다. 따라서 pearson 상관관계는 1보다 작지만, spearman 계수는 1이다.    
* Matthew Correlation
    * phi coefficient라고도 함
    * Binary classification에 사용되는 metric 중 1개
    * -1 ~ 1 사이값
    * $ MCC= {TP \times TN - FP \times FN \over \sqrt {(TP+FP)(TP+FN)(TN+FP)(TN+FN)} } $
        * TP(True Positive): 맞다고 예측(positive)했는데 그 예측이 맞는(true) 것의 개수
        * TN(True Negative): 아니라고 예측(negative)했는데 그 예측이 맞는(true) 것의 개수
        * FP(False Positive): 맞다고 예측(positive)했는데 그 예측이 틀린(false) 것의 개수
        * FN(False Negative): 아니라고 예측(negative)했는데 그 예측이 틀린(false) 것의 개수
    * Matthew Correlationsms 두 벡터의 Pearson Correlation과 동일
        * 즉, 정답 벡터와 예측 벡터가 얼마나 유사한지를 보여주는 metric
        * [참고자료](https://ivoryrabbit.github.io/%EC%88%98%ED%95%99/2021/03/12/%EB%A7%A4%ED%8A%9C-%EC%83%81%EA%B4%80%EA%B3%84%EC%88%98.html)

# 2. Code

```python
def preprocess_function(examples):
    # 두번째 문장이 없는 task의 경우
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # task/metric에 따라 계산 방식이 달라짐
    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer 

# pretrained model setting
model_checkpoint = "distilbert-base-uncased"
batch_size = 16

# tokenizer load
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

# 문장 1, 2에 대한 설정
    # 현재 task 에서는 sentence1_key: sentence, sentence2_key: None
sentence1_key, sentence2_key = task_to_keys[task]
    # caching을 통해 재실행시 다시 결과를 불러오는 것을 방지
    # load_from_cache_file=False 로 새롭게 불러올 수 있음
    # batched=True: muti threading
encoded_dataset = dataset.map(preprocess_function, batched=True)

# model load
    # classification을 위한 head 설정
num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

# metric 설정
metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"
model_name = model_checkpoint.split("/")[-1]

# 훈련을 위한 파라 미터 설정
args = TrainingArguments(
    "test-glue",
    evaluation_strategy = "epoch", # 에포크에 따라 평가 진행
    save_strategy = "epoch", # 에포크에 따라 저장
    learning_rate=2e-5, # 학습률
    per_device_train_batch_size=batch_size, # 훈련 batch size
    per_device_eval_batch_size=batch_size, # 평가 batch size
    num_train_epochs=5, # 학습 epoch
    weight_decay=0.01, # learning_rage scheduler
    load_best_model_at_end=True, # 마지막에 최고 성능 모델 load
    metric_for_best_model=metric_name, # 평가 기준
    push_to_hub=True, # hugging face hub에 올릴건지
    push_to_hub_model_id=f"{model_name}-finetuned-{task}",
)

# 훈련 class 생성
validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"
trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 훈련
trainer.train()

# 평가
trainer.evaluate()

# Hyper parameter search
def model_init():
    return AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

# model을 model_init 함수로 설정
trainer = Trainer(
    model_init=model_init,
    args=args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 가장 성능이 좋은 조합의 hyperparameter 반환
best_run = trainer.hyperparameter_search(n_trials=10, direction="maximize")

# 아래 코드를 이용해 가장 좋았던 파라미터 재 setting 가능
for n, v in best_run.hyperparameters.items():
    setattr(trainer.args, n, v)
```