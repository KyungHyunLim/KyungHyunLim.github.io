---
layout: post
title:  "Huggingface - RobertaForSequenceClassification의 반환값 분석"
date:   2021-10-04 16:06:22
categories: [ML_AI]
use_math: true
---

# 1. huggingface의 pretrained 모델
## 1.1 원본 소스 링크 github
* [링크](https://github.com/huggingface/transformers/blob/master/src/transformers/models/roberta/modeling_roberta.py)
    * 복잡하고...뭐가 굉장히 많습니다.
    * 중요한 부분만 살펴볼까요!

## 1.2 우리가 호출하는 모델의 정체는?
* 먼저 우리는 huggingface의 pretrained 모델을 불러올 때 아래와 같이 사용합니다.

    ```python
    mymodel = RobertaForSequenceClassification.from_pretrained('원하는 pretrained 모델 이름')
    ```

* 굉장히 간단하다. 하지만 그만큼 우리가 custom할 수 있는게 많이 없습니다. 아니, 어떻게 접근해야할지 감이 오지 않는 다고 하는 것이 맞을 것 같습니다. 하지만, mymodel을 출력해보면 뭔가 있숙한 구조가 나오는 것을 알 수 있습니다(너무 길어서 중간에 생략을 좀 했습니다.). 일반적으로 pytorch 모델을 설계하고 출력하면 나오는 형태와 유사합니다. 이제 막 custom해볼 수 있을 것 같은 용기가 솟아나는 순간입니다.

    ```python
    print(mymodel)

    RobertaForSequenceClassification(
        (roberta): RobertaModel(
            (embeddings): RobertaEmbeddings(
            (word_embeddings): Embedding(32000, 768, padding_idx=1)
            (position_embeddings): Embedding(514, 768, padding_idx=1)
            (token_type_embeddings): Embedding(1, 768)
            (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
            )
            (encoder): RobertaEncoder(
            (layer): ModuleList(
                (0): RobertaLayer(
                (attention): RobertaAttention(
                    (self): RobertaSelfAttention(
                    (query): Linear(in_features=768, out_features=768, bias=True)
                    (key): Linear(in_features=768, out_features=768, bias=True)
                    (value): Linear(in_features=768, out_features=768, bias=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                    )
                    (output): RobertaSelfOutput(
                    (dense): Linear(in_features=768, out_features=768, bias=True)
                    (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                    )
                )
                (intermediate): RobertaIntermediate(
                    (dense): Linear(in_features=768, out_features=3072, bias=True)
                )
                (output): RobertaOutput(
                    (dense): Linear(in_features=3072, out_features=768, bias=True)
                    (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                )
                )
                ... (생략)
                (11): RobertaLayer(
                (attention): RobertaAttention(
                    (self): RobertaSelfAttention(
                    (query): Linear(in_features=768, out_features=768, bias=True)
                    (key): Linear(in_features=768, out_features=768, bias=True)
                    (value): Linear(in_features=768, out_features=768, bias=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                    )
                    (output): RobertaSelfOutput(
                    (dense): Linear(in_features=768, out_features=768, bias=True)
                    (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                    )
                )
                (intermediate): RobertaIntermediate(
                    (dense): Linear(in_features=768, out_features=3072, bias=True)
                )
                (output): RobertaOutput(
                    (dense): Linear(in_features=3072, out_features=768, bias=True)
                    (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                )
                )
            )
            )
        )
        (classifier): RobertaClassificationHead(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
            (out_proj): Linear(in_features=768, out_features=30, bias=True)
        )
    )
    ```

# 2. 원본 소스 분석
## 2.1 RobertaForSequenceClassification forward 함수 살펴보기
* 이제는 이 모델이 어떤 출력을 내보낼지 예상할 수 있습니다. 이 모델은 classification을 위한 모델로, 마지막 classifier에서 클래수 개수 만큼의 logits를 출력으로 돌려줍니다. 하지만 조금더 살펴보아야 할 점이 있습니다. forward의 정의를 한번 더 살펴 보도록 해요!
* ((loss,) + output) if loss is not None else output 이 최종 ouput입니다. loss는 사실 왜 벌써 구하는지 모르겠다. 이건 전체 pipe line을 뜯어봐야 알 것 같습니다.

    ```python
    class RobertaForSequenceClassification(RobertaPreTrainedModel):
        _keys_to_ignore_on_load_missing = [r"position_ids"]

        def __init__(self, config):
            super().__init__(config)
            self.num_labels = config.num_labels
            self.config = config

            self.roberta = RobertaModel(config, add_pooling_layer=False)
            self.classifier = RobertaClassificationHead(config)

            self.init_weights()

        @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
        @add_code_sample_docstrings(
            tokenizer_class=_TOKENIZER_FOR_DOC,
            checkpoint=_CHECKPOINT_FOR_DOC,
            output_type=SequenceClassifierOutput,
            config_class=_CONFIG_FOR_DOC,
        )

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        ):
            r"""
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
                config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
                If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
            """
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            outputs = self.roberta(
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
            sequence_output = outputs[0]
            logits = self.classifier(sequence_output)

            loss = None
            if labels is not None:
                if self.config.problem_type is None:
                    if self.num_labels == 1:
                        self.config.problem_type = "regression"
                    elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                        self.config.problem_type = "single_label_classification"
                    else:
                        self.config.problem_type = "multi_label_classification"

                if self.config.problem_type == "regression":
                    loss_fct = MSELoss()
                    if self.num_labels == 1:
                        loss = loss_fct(logits.squeeze(), labels.squeeze())
                    else:
                        loss = loss_fct(logits, labels)
                elif self.config.problem_type == "single_label_classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                elif self.config.problem_type == "multi_label_classification":
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(logits, labels)

            if not return_dict:
                output = (logits,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output

            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
    ```

* output의 0번째 index는 [CLS] token의 임베딩 벡터입니다. 이것을 classifier에 전달해 logits를 계산합니다. 즉, RobertaForSequenceClassification 모델의 forward 계산의 첫 인덱스에 들어있는 정보는 logits 입니다. (각 클래스일 확률 값) 

    ```python
    sequence_output = outputs[0]   
    logits = self.classifier(sequence_output)
    ```

* 이제 outputs[2:]의 의미를 알아보도록 해요! 이 친구는 self.roberta 안에 숨어있습니다. 그리고 self.roberta는 RobertaModel class를 의미합니다.
* RobertaModel forward는 역시 너무깁니다. 일부분만 가져와 보자면, encoder_ouputs의 첫번째 인덱스의 값과, sequence_output을 self.pooler에 적용한 값을 튜플로 묶고, 그 이후에 encoder_outputs를 더해주고 있습니다. 무슨 의미인지 아직 감이 오지 않습니다.

    ```python
    embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
    )
    encoder_outputs = self.encoder(
        embedding_output,
        attention_mask=extended_attention_mask,
        head_mask=head_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_extended_attention_mask,
        past_key_values=past_key_values,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    sequence_output = encoder_outputs[0]
    pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

    if not return_dict:
        return (sequence_output, pooled_output) + encoder_outputs[1:]
    ```

* 차근차근 다가가 보면, 우선 우리는 encoder의 입력인 embedding_output을 이해해야 합니다.
    * position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)로 포지션 ids도 자동으로 만들어주고
    * token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)도 자동으로 만들어 줍니다.
    * inputs_embeds = self.word_embeddings(input_ids) 로 단어 임베딩 벡터를 만들고,
    * token_type_embeddings = self.token_type_embeddings(token_type_ids) 문장 순서 임베딩 벡터를 만들고
    * embeddings = inputs_embeds + token_type_embeddings 두개를 더해줍니다.
    * position_embeddings는 default가 absolute로 설정되어 있는 것 같다. 이것도 embeddings에 더해줍니다.
    * Layer norm과 dropout을 거쳐 반환을 합니다.
    * 즉, embedding_output은 transformer 구조에 들어가기 위해 token을 벡터로 바꾼것!!

    ```python
    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    ```

* 이제 encoder입니다! 매우 깁니다... 계산과정은 복잡하니 return값만 확인해보겠습니다!
    * hidden_states: 일단 입력으로 받고 시작하는데, 이게 방금 살펴본 embedding입니다. 이 친구를 robertlayer에 지정된 layer들에 넣어 layer_outputs을 얻습니다. 그리고 0번 인덱스를 다시 hidden_states에 할당해줍니다.
        * 0번이 embedding vector 값
        * 1번이 self_attentions 값
        * 2번이 cross_attentions 값 이 되는 것 같습니다.
    * 이것들을 반환해줍니다.

    ```python
    return tuple(
        v
        for v in [
            hidden_states,
            next_decoder_cache,
            all_hidden_states,
            all_self_attentions,
            all_cross_attentions,
        ]
        if v is not None
    )
    ```

* (sequence_output, pooled_output) + encoder_outputs[1:] 로 돌아와서 보면, encoder_outputs[1:]는 next_decoder_cache, all_hidden_states, all_self_attentions, all_cross_attention를 의미합니다.
* 다시 정리해보자면, RobertaModel class의 output은 (sequence_output, pooled_output) + encoder_outputs[1:] 입니다.
* 즉, classification 클래스에서, outputs = [embedding vector, pooled_output, next_decoder_cache, all_hidden_states, all_self_attentions, all_cross_attention] 가 됩니다.
    * 최종 output은 아래와 같기 때문에, embedding vector로 logits를 구하고, next_decoder_cache, all_hidden_states, all_self_attentions, all_cross_attention를 튜플로 붙인 형태가 됩니다. 거기에 loss까지 계산이 됬으면
    * loss, logits, next_decoder_cache, all_hidden_states, all_self_attentions, all_cross_attention 가 최종 형태가 됩니다.

    ```python
    output = (logits,) + outputs[2:]
    return ((loss,) + output) if loss is not None else output
    ```

## 2.2 최종 형태의 의미
* loss, logits, next_decoder_cache, all_hidden_states, all_self_attentions, all_cross_attention 가 최종형태입니다. 왜, output의 첫번째 인덱스가 각 클래스일 확률인지를 이해할 수 있었습니다. 
* 물론 이외에 더 복잡하고 연관된 사항들이 많기 때문에 아직은 갈길이 멀다고 생각합니다. 다음번 포스팀에서는 조금더 세세한 의미를 알아볼 수 있도록 노력해보겠습니다!

```python
# 실제 출력
SequenceClassifierOutput(logits=tensor([[ 0.0800,  0.1805,  0.0620,  ..., -0.1257, -0.1650, -0.3279],
        [ 0.1403,  0.1399,  0.0256,  ..., -0.1489, -0.0258, -0.4063],
        [ 0.1422,  0.1240,  0.1042,  ...,  0.0295, -0.2008, -0.4127],
        ...,
        [-0.0334,  0.2673, -0.0113,  ..., -0.0961, -0.0519, -0.4188],
        [ 0.0981,  0.1145, -0.0784,  ..., -0.0449, -0.1429, -0.3333],
        [ 0.0175,  0.2102, -0.0617,  ..., -0.1251, -0.0935, -0.4533]],
       device='cuda:0', grad_fn=<AddmmBackward>), hidden_states=None, attentions=None)
```