---
layout: post
title:  "torcn.nn.Embedding 알아보기"
date:   2021-10-06 13:15:12
categories: [Pytorch, ML_AI]
use_math: true
---

# 1. 모듈 알아보기
## 1.1 공식 Documentation
* pytorch 1.9.1 ver
* 입력 받는 파라 미터 알아 보기

    ```python
    torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None, device=None, dtype=None)
    ```

    * num_embeddings(int): 임베딩을 위한 사전의 크기 (유니크한 값의 개수)
    * embedding_dim(int): 각 값의 임베딩 벡터 크기
    * padding_idx(int, optional): 지정하면, padding_idx는 gradient 값에 기여하지 않음! 즉, padding_idx는 학습 중에 업데이트 되지 않는다!
    * max_norm(float, optional): 주어진 max_norm값 보다 큰 값은 다시 노멀라이즈 함
    * norm_type(float, optional): p-norm 의 p

## 1.2 Documentation Examples
* Embedding layer 선언

    ```python
    import torch.nn as nn
    embedding = nn.Embedding(10, 3)
    ```

* 활용해보기

    ```python
    import torch
    input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
    embedding(input)
    
    ----------
    OUTPUT
    ----------
    tensor([
        [[-0.0251, -1.6902,  0.7172],
        [-0.6431,  0.0748,  0.6969],
        [ 1.4970,  1.3448, -0.9685],
        [-0.3677, -2.7265, -0.1685]],

        [[ 1.4970,  1.3448, -0.9685],
        [ 0.4362, -0.4004,  0.9400],
        [-0.6431,  0.0748,  0.6969],
        [ 0.9124, -2.3616,  1.1151]]])
    ```

## 1.3 실제 사용해보기
* num_embeddings가 2인데 그 이상의 값들을 넣으면 에러가 뜬다. 2 이면 입력 벡터는 0과 1로만 이루어져 있어야 한다.

    ```python
    import torch
    import torch.nn as nn

    embedding = nn.Embedding(2, 5)
    input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])

    ----------
    OUTPUT
    ----------
    IndexError: index out of range in self
    ```

* embedding 값 확인해보기  
    ![](/assets/image/pytorch/em1.PNG)
    * 당연한 얘기지만 모듈을 재선하면 값이 달라진다. weigth의 초기 값에 의해 결정되는 것 같다.
    * 너무 작은 값이 나올 수 도 있는것 같다. 주의하면 좋을 것 같다.