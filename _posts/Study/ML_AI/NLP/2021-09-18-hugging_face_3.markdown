---
layout: post
title:  "Huggingface- Chapter 3. Fine-tunning"
date:   2021-09-18 14:15:22
categories: [NLP, ML_AI]
use_math: true
---

# Chapter 3. Fine-tuning a pretrained model
## 1. Pre-processing
* [Dataset](https://huggingface.co/datasets)

    ```python
    # Load dataset
    from datasets import load_dataset

    raw_datasets = load_dataset("glue", "mrpc")

    # Load tokenizer
    from transformers import AutoTokenizer

    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # Example of multiple sequence
    inputs = tokenizer("This is the first sentence.", "This is the second one.")
    print(inputs)

    # dataset tokenizer 적용
    tokenized_dataset = tokenizer(
        raw_datasets["train"]["sentence1"],
        raw_datasets["train"]["sentence2"],
        padding=True,
        truncation=True,
    )

    ----------
    OUTPUT
    ----------
    { 
    'input_ids': [101, 2023, 2003, 1996, 2034, 6251, 1012, 102, 2023, 2003, 1996, 2117, 2028, 1012, 102],
    'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    }
    ```

* Collate function
    * Batch내 적절하게 pad 추가
    * DataCollatorWithPadding

    ```python
    from transformers import DataCollatorWithPadding
    # Tokenizer에 collate function 추가
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    samples = tokenized_datasets["train"][:8]

    # tokenizer 적용
    batch = data_collator(samples)
    # 결과 확인하기
    {k: v.shape for k, v in batch.items()}

    ----------
    OUTPUT
    ----------
    {'attention_mask': torch.Size([8, 67]),
    'input_ids': torch.Size([8, 67]),
    'token_type_ids': torch.Size([8, 67]),
    'labels': torch.Size([8])}
    ```

## 2. Fine-tuning a model with the Trainer API
* Trainer class
    * train arg 설정
        * training 및 evaluation을 위한 Custom Trainer 정의

    ```python
    from transformers import TrainingArguments

    training_args = TrainingArguments("test-trainer")
    ```

* Model 불러오기
    * Pretrained BERT 모델에서 기존의 head 버리고, num_labels에 맞는 head 자동 생성

    ```python
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    ```

* Train
    * 아래 예시는 default
    * evaluation_strategy: steps나 epoch 단위 부여 가능
    * compute_metrics: 지정 가능, 기본은 loss 만 출력
    * fp16 = True로 mixed-precision training 적용 가능

    ```python
    from transformers import Trainer

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator, # 위에서 tokenizer에 DataCollatorWithPadding 를 적용했으면 생략 가능
        tokenizer=tokenizer,
    )
    # strat training
    trainer.train()
    ```
* Evaluation
    * Trainer.predict를 사용해 예측
        * logit 반환

    ```python
    predictions = trainer.predict(tokenized_datasets["validation"])
    print(predictions.predictions.shape, predictions.label_ids.shape)   

    preds = np.argmax(predictions.predictions, axis=-1)
    ----------
    OUTPUT
    ----------
    (408, 2) (408,)
    ```

    * load_metric 활용

    ```python
    from datasets import load_metric

    metric = load_metric("glue", "mrpc")
    metric.compute(predictions=preds, references=predictions.label_ids)
    ----------
    OUTPUT
    ----------
    {'accuracy': 0.8578431372549019, 'f1': 0.8996539792387542}
    ```

    * Train 함수에 적용하기

    ```python
    def compute_metrics(eval_preds):
        metric = load_metric("glue", "mrpc")
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics # Evaluation 방법 정의
    )
    ```

## 3. Full training

```python
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

# 데이터셋 불러오기
raw_datasets = load_dataset("glue", "mrpc")
# tokenizer 불러오기
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

# batched=True 여러 문장을 동시에 처리, 처리속도 향상
# The results of the function are cached
# It does not load the whole dataset into memory, saving the results as soon as one element is processed.

 
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
# 최종 tokenizer 설정
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 모델에서 사용하지 않는 column 제거
tokenized_datasets = tokenized_datasets.remove_columns(
    ["sentence1", "sentence2", "idx"]
)
# 모델에서 사용하는 이름으로 변경 laber -> labels
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
# torch tensor로 변환
tokenized_datasets.set_format("torch")
# ['attention_mask', 'input_ids', 'labels', 'token_type_ids']
tokenized_datasets["train"].column_names 

# dataloader 정의
train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

# model 정의, Optimizer 정의, scheduler 정의
from transformers import AutoModelForSequenceClassification
from transformers import AdamW
from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
optimizer = AdamW(model.parameters(), lr=5e-5)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# Training
progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch) # forward 연산
        loss = outputs.loss    # loss 계산
        loss.backward()        # grad 계산
        
        optimizer.step()       # optimizer 적용
        lr_scheduler.step()    # scheduler 적용
        optimizer.zero_grad()  # optimizer 초기화
        progress_bar.update(1) # 진행바 표시

# Evaluation
from datasets import load_metric

metric= load_metric("glue", "mrpc") # metric load
model.eval() # eval 모드로 변경
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad(): # grad 계산 없이 진행
        outputs = model(**batch) # forward 연산
    
    logits = outputs.logits # 샘플별 각 class에 대한 확률값
    predictions = torch.argmax(logits, dim=-1) # 높은 확률 값으로 class 결정
    # 메트릭 적용
    metric.add_batch(predictions=predictions, references=batch["labels"])

# 최종 메트릭 계산
metric.compute()
```

* accelerator를 활용해 더 빠르게 학습

```python
from accelerate import Accelerator

accelerator = Accelerator()

train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
     train_dataloader, eval_dataloader, model, optimizer
)

...

accelerator.backward(loss)
```