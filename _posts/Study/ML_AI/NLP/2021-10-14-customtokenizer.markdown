---
layout: post
title:  "Custom data를 이용해 WordPiece 방식으로 vocab 생성 및 BertTokenizer 훈련하기"
date:   2021-10-14 22:07:22
categories: [NLP, ML_AI]
use_math: true
---

# 1. BertWordPieceTokenizer 활용하기!
## 1.1 Data 준비하기
* Data는 AI hub에서 제공해주는 대화데이터를 활용하였습니다.
    * 대화 요약을 위한 데이터셋
    * [데이터 링크](https://aihub.or.kr/aidata/30714)

    ```python
    from glob import glob
    train_path_list = glob('./Train/*')

    # str 리스트를 만들어 주시면 됩니다.
    # E.g. ['집에서 밥을 해먹어야 해서 엄마 카드(엄카)를 가지고 장을 보러 간다.', '실바니안은 코스트코에 가서 사는 것보다 인터넷으로 사는 게 더 싸다.']
    train_json_list = make_dataset_list(train_path_list)
    train_data= make_set_as_df(train_json_list)
    encoder_input_train = make_tokenizer_input(train_data)
    ```

## 1.2 Hugging face tokenizers 라이브러리 활용하기
* BertWordPieceTokenizer 클래스를 사용해 새로운 vocabulary 사전을 만들어 보겠습니다.

```python
from tokenizers import BertWordPieceTokenizer 

# BertWordPieceTokenizer 선언
    # None 자리는 원래 vocabulary 파일 경로가 들어가는 자리입니다.
    # 하지만 저희는 새로운 vocabulary를 만들것이기 때문에 None으로 만듭니다.
    # 그 외에는 자유롭게 변경하셔도 되는 부분입니다.
tokenizer = BertWordPieceTokenizer(
    None,
    clean_text=True,
    handle_chinese_chars=True,
    strip_accents=False, # Must be False if cased model
    lowercase=False,
    wordpieces_prefix="##",
)
# 훈련하기
    # .train을 쓰면 파일 경로를 주어서 학습할 수 있고
    # .train_from_iterator를 쓰면은 List[str] 형태를 주어 학습할 수 있습니다.
    # special token과 vocab size를 정하고 학습합니다.
tokenizer.train_from_iterator(
    encoder_input_train,
    vocab_size=36000,
    min_frequency=2,
    show_progress=True,
    special_tokens = ["[PAD]", "[CLS]", "[UNK]", "[SEP]", "[MASK]"],
    wordpieces_prefix="##",
)
```

## 1.3 Vocab 추출하기
* BertWordPieceTokenizer 자체로 사용하기에는 Hugging face의 transfomers의 기능을 온전히 이용하기에 무리가 있습니다.
* 그래서 새롭게 만든 vocabulary를 활용해 BertTokenizer를 만들어 보겠습니다.

```python
vocab_dir = './vocab/vocab.txt'

# tokenizer에서 vocab 가져오기
vocab = tokenizer.get_vocab()

# index 번호에 맞게 정렬
vocabulary = [[v, k] for k, v in vocab.items()]
vocabulary = sorted(vocabulary)
vocabulary = list(np.array(vocabulary)[:, 1])

# vocabulary 저장
with open(vocab_dir, 'w+') as lf:
    lf.write('\n'.join(vocabulary))
```

# 2. BertTokenizer 만들기
## 2.1 BertTokenizer
* 두가지 방식으로 BertTokenizer에 새롭게 만든 vocabulary를 집어 넣을 수 있습니다.
* 주의할 점은 "do_basic_tokenize=False" 이것을 해주어야지만 WordPiece 방식의 tokenizer를 사용할 수 있다는 것입니다.
    * 왜냐하면, BertTokenizer의 구현부를 살펴보면 기본 tokenize를 사용하면 띄어쓰기를 기반으로 tokenizing을 하는 함수가 실행 되기 때문입니다.
    * [참고](https://github.com/huggingface/transformers/blob/cc360649606f1a0105c9d465a2522a454746894f/src/transformers/models/bert/tokenization_bert.py#L201)

```python
from transformers import BertTokenizer

# 1.
bert_tokenizer = BertTokenizer('./vocab/', do_basic_tokenize=False)

# 2.
bert_tokenizer = BertTokenizer.from_pretrained('./vocab/', do_basic_tokenize=False)
```

## 2.2 결과 확인해보기
![](/assets/image/)
* 잘 tokenization 된것을 확인 할 수 있습니다.
* [UNK] token을 줄이기 위해서 최적의 vocab size나 학습 예문을 늘리는 방향도 필요할 것 같습니다.

## 2.3 이외의 Tokenizer에 적용할려면?
* Bart tokenizer는 roberta tokenizer를 상속받고, roberta tokenizer는 gpt-2의 tokenizer를 상속받고 있습니다.
* 이들의 특징은 merges 파일이 필요하다는 것인데, WordPiece 방식을 사용하는 과정에서 합쳐지는 과정을 의미하는 것 같습니다.
* 이부분을 얻을 수 있는 방법, 또는 새롭게 구현을 한다면 다시 포스팅을 수정하도록 하겠습니다!