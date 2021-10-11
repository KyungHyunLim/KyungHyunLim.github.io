---
layout: post
title:  "Hugging face - QA model classification head 분석해보기"
date:   2021-10-11 18:00:22
categories: [NLP, ML_AI]
use_math: true
---

# 1. QA란?
* Question Answering의 줄임말로, 어떤 내용이 담긴 본문과 질문을 주면, 질문에 대한 답을 주는 것이다.
* E.g.
    * 본문:  [출처](https://ko.wikipedia.org/wiki/%EC%9C%84%ED%82%A4)
    위키는 간단한 마크업 언어와 웹 브라우저를 이용, 함께 문서를 작성하는 공동체를 가능케 한다. 위키 웹사이트의 한 문서는 "위키 문서"라 부르며, 하이퍼링크로 서로 연결된 전체 문서를 "위키"라 한다. 위키는 본질적으로 정보를 만들고, 찾아보고, 검색하기 위한 데이터베이스다. 위키는 비선형적인, 진화하는, 복잡하게 얽힌 문서, 토론, 상호 작용을 할 수 있게 돕는다.
    * 질문: 위키는 어떤 언어를 이용하는가?
    * 답: 마크업 언어

# 2. 코드분석
## 2.1 init 부분
* BertModel을 통해 얻은 임배딩 벡터를 이용한다.
* 정답을 찾기위한 classification head는 self.qa_outputs 인데 생각보다 간단한 형태이다.
    * hidden_size $\rightarrow$ num_labels
    * 여기서 num_labels는 정답을 찾아야하는 본문의 토큰 개수일 것이다.

    ```python
    def __init__(self, config):
            super().__init__(config)
            self.num_labels = config.num_labels

            self.bert = BertModel(config, add_pooling_layer=False)
            self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

            self.init_weights()
    ```

## 2.2 foward 정의 부분
* SequenceClassification을 위한 모델과는 다르게 추가된 입력들이 있다.
    * start_positions, end_position이다.
    * answer의 token의 시작점과 끝점을 의미한다.

    ```python
    r"""
    start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
        Labels for position (index) of the start of the labelled span for computing the token classification loss.
        Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
        sequence are not taken into account for computing the loss.
    end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
        Labels for position (index) of the end of the labelled span for computing the token classification loss.
        Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
        sequence are not taken into account for computing the loss.
    """
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        ):
    ```

## 2.3 forward 구현 부분
* BERT 몸체의 임베딩 결과를 받아오는 부분은 어떠한 task 모델이든 동일하다.

    ```python
    outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    ```
* BERT로 부터 받아온 outputs의 0번째 index값을 사용한다.
    * 이전에 roberta를 분석했을 때도 알아보았지만, 이것은 입력되 토큰들의 임베딩 벡터이다.
    * 즉, batch가 4이고, 30개의 토큰이 입력으로 들어왔고, 임베딩 벡터의 차원이 50이면,
        * (4, 30, 50) size의 tensor가 될 것이다.
    * 이것을 qa_ouputs에 통과 시킨다. 
        * 그럼 결과로 (4, 30, 2) 크기의 logit이 나올 것이다.
        * 이것은 0번은 start_index일 확률, 1번은 end_index일 확률을 의미한다.
        * 즉, dim=1 기준으로 start_index/end_index일 확률을 split 한다.
        * 그리고 쓸모 없는 차원을 줄어 확률 값들을 1차원 배열(텐서)로 만든다.

    ```python
    sequence_output = outputs[0]

    logits = self.qa_outputs(sequence_output)
    start_logits, end_logits = logits.split(1, dim=-1)
    start_logits = start_logits.squeeze(-1).contiguous()
    end_logits = end_logits.squeeze(-1).contiguous()
    ```

* 입력으로 start_positions/end_positions이 주어졌으면, 값을 비교해 loss 계산을 한다.
    * 가끔 정답이 주어진 본문 밖에 있는 경우가 있을 수 있다. (보통 일정 길이로 잘라서 입력으로 주어지기 때문에), 이부분을 무시하기 위한 코드가 들어가 있다. 이부분은 전처리에 따라 작동할 수 도 안할 수도 있다.
    * 그리고 각각 CrossEntropyLoss를 이용해 start_loss, end_loss를 계산하고, 두 loss의 평균을 total loss로 할당한다.

    ```python
    total_loss = None
    if start_positions is not None and end_positions is not None:
        # If we are on multi-GPU, split add a dimension
        if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)
        # sometimes the start/end positions are outside our model inputs, we ignore these terms
        ignored_index = start_logits.size(1)
        start_positions = start_positions.clamp(0, ignored_index)
        end_positions = end_positions.clamp(0, ignored_index)

        loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2
    ```

* 이제 반환하는 부분이다. 
    * ouput을 (start_logits, end_logits)의 형태로 만든다. 그 이후에 bert에서 얻은 여러 output들을 붙여준다.
    * ((total_loss,) + output)를 최종 형태로 반환해준다.
    * 정답이 없을 경우에는 loss 계산이 되지 않기 때문에 output만 반환 할 수 있도록 조건문을 달아 두었다.

    ```python
     if not return_dict:
        output = (start_logits, end_logits) + outputs[2:]
        return ((total_loss,) + output) if total_loss is not None else output
    ```

# 3. 정리
* 내부에서 BERT의 임베딩 벡터를 받아 각 토큰에 대해, start index/end index일 확률을 계산하는 듯 보인다. 이 확률값들의 최대값을 찾으면 그 토큰이 가장 가능성 높은 시작점/끝점을 의미하게 된다. 아직 qa_outputs부분이 제대로 이해되지 않는 것 같다. 더 알아보고 오류가 있으면 수정해야 할 것 같다.

# 4. 참조
* [original src](https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_bert.py)