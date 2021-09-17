---
layout: post
title:  "Huggingface- Chapter 1"
date:   2021-09-17 10:15:22
categories: [NLP, ML_AI]
use_math: true
---

# 1. Huggingface
* 자연어처리 모델들을 지원해주는 라이브러리
* [링크](https://huggingface.co/)
* [Git](https://github.com/huggingface)

# 2. Chapter 1
## 2.1 pipeline
* 기본적으로는 영어로 된 감정 분석을 위해 미세 조정된 특정 사전 훈련된 모델을 선택
    * 그런데 아래 경우처럼 한글도 어느정도 분류해주는 신기함을 가짐

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier([
    "I've been waiting for a HuggingFace course my whole life.", 
    "사랑해!",
    "죽을래?"
])
----------
OUTPUT
----------
[{'label': 'POSITIVE', 'score': 0.9598048329353333},
 {'label': 'POSITIVE', 'score': 0.9939879179000854},
 {'label': 'NEGATIVE', 'score': 0.9216691255569458}]
```
* 3가지 단계 자동 처리
    1. Sentence preprocessing
    2. Model에 preprocessed stentence 넣어서 추론
    3. Post-processing, Model의 prediction 결과 정리
* 현재 이용가능한 pipelines
    * feature-extraction (get the vector representation of a text)
    * fill-mask
        * 추론시, top_k 설정 -> 몇개를 보여줄지
    * ner (named entity recognition)
        * 모델 정의시 grouped_entities=True, 같은 entity끼리 grouping
    * question-answering
    * sentiment-analysis
    * summarization
        * max_length or a min_length 설정 가능
    * text-generation
        * max_length or a min_length 설정 가능
    * translation
    * zero-shot-classification
    
    ```python
    # model list = https://huggingface.co/models?pipeline_tag=zero-shot-classification
    # model list = https://huggingface.co/models?pipeline_tag=text-generation
    classifier = pipeline("zero-shot-classification", model='typeform/distilbert-base-uncased-mnli')
    classifier(
        "This is a course about the Transformers library",
        candidate_labels=["education", "politics", "business"],
    )
    ----------
    OUTPUT
    ----------
    # Default model
    {'sequence': 'This is a course about the Transformers library',
    'labels': ['education', 'business', 'politics'],
    'scores': [0.844597339630127, 0.11197531968355179, 0.04342734441161156]}
    
    # typeform/distilbert-base-uncased-mnli
    {'sequence': 'This is a course about the Transformers library',
    'labels': ['education', 'politics', 'business'],
    'scores': [0.43294695019721985, 0.3514440059661865, 0.21560902893543243]}
    ```
## 2.2 Transformer introduction
* 크게 3가지 카테고리
    * GPT-like (also called auto-regressive Transformer models)
    * BERT-like (also called auto-encoding Transformer models)
    * BART/T5-like (also called sequence-to-sequence Transformer models)
* 기본적인 구조
    * Encoder: Input을 받아, 그것에 대한 representaion 생성
    * Decoder: Encoder의 representation을 이용해 target sequence 생성
* 모델 구조에 따른 활용도    
    * Encoder only model
        * Input을 이해하는 것이 필요한 task에 좋음
        * E.g. sentence classification, named entity recognition.
        * Models
            * [ALBERT](https://huggingface.co/transformers/model_doc/albert.html)
            * [BERT](https://huggingface.co/transformers/model_doc/bert.html)
            * [DistillBERT](https://huggingface.co/transformers/model_doc/distilbert.html)
            * [ELECTRA](https://huggingface.co/transformers/model_doc/electra.html)
            * [RoBERTa](https://huggingface.co/transformers/model_doc/roberta.html)
    * Decoder only model
        * 생성하는 task에 좋음
        * E.g. text generation.
        * Models
            * [CTRL](https://huggingface.co/transformers/model_doc/ctrl.html)
            * [GPT](https://huggingface.co/transformers/model_doc/gpt.html)
            * [GPT-2](https://huggingface.co/transformers/model_doc/gpt2.html)
            * [Transformer XL](https://huggingface.co/transformers/model_doc/transformerxl.html)
    * Encoder-decoder models or sequence-to-sequence models
        * Input이 필요한 생성 task에 좋음
        * E.g. translation, summarization.
        * Models
            * [BART](https://huggingface.co/transformers/model_doc/bart.html)
            * [mBART](https://huggingface.co/transformers/model_doc/mbart.html)
            * [Marian](https://huggingface.co/transformers/model_doc/marian.html)
            * [T5](https://huggingface.co/transformers/model_doc/t5.html)
* 용어 이해
    * Architecture: 모델의 뼈대 - 각 layer와 모델에서 일어나는 operation 정의
    * Checkpoints: Architecture에 로드할 수 있는 weights
    * Model: “architecture” or “checkpoint” 두가지를 모두 의미
* Bias and limitations
    * Pre-trianed된 모델이 sexist, racist, or homophobic content를 만들 수 도 있음
    * E.g
        * Top-5 단어 후보들이 주로 성별에 치우쳐져 있음

    ```python
    unmasker = pipeline("fill-mask", model="bert-base-uncased")
    result = unmasker("This man works as a [MASK].")
    print([r["token_str"] for r in result])

    result = unmasker("This woman works as a [MASK].")
    print([r["token_str"] for r in result])
    ----------
    OUTPUT
    ----------
    ['lawyer', 'carpenter', 'doctor', 'waiter', 'mechanic']
    ['nurse', 'waitress', 'teacher', 'maid', 'prostitute']
    ```