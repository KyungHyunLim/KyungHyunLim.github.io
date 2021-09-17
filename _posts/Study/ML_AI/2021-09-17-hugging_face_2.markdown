---
layout: post
title:  "Huggingface- Chapter 2"
date:   2021-09-17 10:15:22
categories: [NLP, ML_AI]
use_math: true
---

# Chapter 2. Using Transformers
## 1. Tokenizer
* Transformer 모델이 처리할 수 있도록 문장을 전처리
    * Split, word, subword, symbol 단위 => token
    * token과 integer 맵핑
    * 모델에게 유용할 수 있는 추가적인 인풋을 더해줌
* AutoTokenizer class
    * 다양한 pretrained 모델을 위한 tokenizer들
    * Default: distilbert-base-uncased-finetuned-sst-2-english in sentiment-analysis

    ```python
    from transformers import AutoTokenizer

    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.", 
    "I hate this so much!",
    ]

    inputs = tokenizer(padding=True, truncation=True, return_tensors="pt")
    print(inputs)

    ----------
    OUTPUT
    ----------
    {'input_ids': tensor([
        [ 101, 1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012, 102],
        [ 101, 1045, 5223, 2023, 2061, 2172, 999, 102, 0, 0, 0, 0, 0, 0, 0, 0]]), 
    'attention_mask': tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])}
    ```

## 2. Pretrained Model
* AutoModel class
    * tokenizer와 같이 pretrained model을 다운로드
    * Batch size: 한번에 처리하는 sequence 수 - 2
    * Sequence length: sequence의 representation length - 16
    * Hidden size: 각 input의 vector 차원 - 768

    ```python
    from transformers import AutoModel

    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    model = AutoModel.from_pretrained(checkpoint)

    outputs = model(**inputs)
    print(outputs.last_hidden_state.shape)

    ----------
    OUTPUT
    ----------
    torch.Size([2, 16, 768])
    ```
* AutoModel + Task
    * Model (retrieve the hidden states)
    * ForCausalLM
    * ForMaskedLM
    * ForMultipleChoice
    * ForQuestionAnswering
    * ForSequenceClassification
    * ForTokenClassification

    ```python
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    outputs = model(**inputs)

    print(outputs.logits.shape)
    print(outputs.logits)

    ----------
    OUTPUT
    ----------
    torch.Size([2, 2])
    tensor([[-1.5607,  1.6123],
        [ 4.1692, -3.3464]], grad_fn=<AddmmBackward>)
    ```

## 3. Models
* Config file
    * 모델 구성을 위한 많은 parameter 존재

    ```python
    BertConfig {
    [...]
    "hidden_size": 768,
    "intermediate_size": 3072,
    "max_position_embeddings": 512,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    [...]
    }
    ```
* Python 
    * 위에서 정의한 Config를 바탕으로 모델 생성
    * Model is randomly initialized!

    ```python
    from transformers import BertConfig, BertModel

    # Building the config
    config = BertConfig()
    # Building the model from the config
    model = BertModel(config)

    # 간단하게 pretrain 모델 불러오기
    # ~/.cache/huggingface/transformers. [다운로드 위치]
    # https://huggingface.co/models?filter=bert 불러올수 있는 모델리스트
    model = BertModel.from_pretrained("bert-base-cased")
    ```
* Save model
    * 간단하게 저장 가능
    * config.json pytorch_model.bin 두개의 파일 저장
        * config.json: 모델의 구조
        * pytorch_model.bin: state dictionary

    ```python
    model.save_pretrained("directory_on_my_computer")
    ```
* Inference
    * torch와 동일하게 사용가능
    * 다양한 arg를 받을 때는, IDs가 필요

    ```python
    import torch

    model_inputs = torch.tensor(encoded_sequences)
    output = model(model_inputs)
    ```

## 4. Tokenizers
* 대표적인 알고리즘
    * Byte-level BPE, as used in GPT-2
    * WordPiece, as used in BERT
    * SentencePiece or Unigram, as used in several multilingual models

    ```python
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    # Encoding
    sequence = "Using a Transformer network is simple"
    tokens = tokenizer.tokenize(sequence)
    # Decoding
    decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
    print(decoded_string)
    ```

## 5. Handling multiple sequences
* Tensorflow의 입력은 multiple sequence
* Batch를 이용해 multiple sequence 구성
    * 다른 길이를 pad를 통해 맞추어 주어야 함

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load tokenizer
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# Load model
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

# apply tokenizer
tokens = tokenizer.tokenize(sequence)
# token to index
ids = tokenizer.convert_tokens_to_ids(tokens)

#################################################################
# This line will fail.
    # Transformers models은 기본적으로 여러 문장을 입력으로 기대
input_ids = torch.tensor(ids)

# This is ok.
input_ids = torch.tensor([ids])
#################################################################
output = model(input_ids)
```
* Padding 시 주의점  
    * padding 한 문장의 loggit 값이 달라진다.

    ```python
    sequence1_ids = [[200, 200, 200]] # 문장 1
    sequence2_ids = [[200, 200]] # 문장 2
    batched_ids = [[200, 200, 200], [200, 200, tokenizer.pad_token_id]]  # Batch

    print(model(torch.tensor(sequence1_ids)).logits)
    print(model(torch.tensor(sequence2_ids)).logits)
    print(model(torch.tensor(batched_ids)).logits)

    ----------
    OUTPUT
    ----------
    tensor([[ 1.5694, -1.3895]], grad_fn=<AddmmBackward>)
    tensor([[ 0.5803, -0.4125]], grad_fn=<AddmmBackward>)
    tensor([[ 1.5694, -1.3895],
            [ 1.3373, -1.2163]], grad_fn=<AddmmBackward>)
    ```

    * Attention mask를 통해 pad 토큰을 무시하게 해주어야 한다.

    ```python
    batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id]
    ]

    attention_mask = [
    [1, 1, 1],
    [1, 1, 0]
    ]
    outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
    
    ----------
    OUTPUT
    ----------
    tensor([[ 1.5694, -1.3895],
            [ 0.5803, -0.4125]], grad_fn=<AddmmBackward>)
    ```

