---
layout: post
title:  "Optimization"
date:   2021-08-10 11:32:22
categories: [ML_AI]
use_math: true
---

## 1. 기본
### 1.1 Gradient Desent
* First-order iterative optimization algorithm for finding a local minimum

### 1.2 최적화 주요 개념!
* Generalization
  * 일반화 성능? 
    학습에 따라 Training error는 줄어들지만 학습을 반복할 수록 Test error는 다시 증가 하게된다. 즉, 학습데이터의 일반화 성능이 좋다고, 좋은 모델이라고 할수 없다.
* Under-fitting vs Over-fitting
  ![](/assets/image/ML_AI/opt_1.png)
  * Under-fitting
    모델이 너무 조금 훈련되서 학습데이터도 잘 맞추지 못하는 상황
  * Over-fitting
    학습데이터는 다 맞추지만, 너무 많이 학습되어 오히려 평가데이터에서는 성능이 떨어지는 상황
  * 저 사이 어딘가를 찾아야 한다!
* Cross validation
  * K-fold validation
  * Train data를 K개의 그룹으로 나누어
    * k-1개는 훈련데이터
    * 나머지 1개를 검증데이터로 사용하는 것
* Bais-variance tradeoff
  ![](/assets/image/ML_AI/opt_2.png)
  * Variance: 출력이 일관되게 나오는지?
  * Bias: 평균과 벗어나는 정도
  * $Given D={(x_i,t_i)}_i^N, where t=f(x)+\epsilon \ and \epsilon \sim N(0, \sigma^2)$
    cost를 최소화하는 것은 $bias^2$ , variance, noise 3가지를 줄이는 것과 동일. 이때 bias를 최소화하면 variance는 높아질 가능성이 크고, 반대의 경우도 동일하기 때문에 trade off를 최소화 하는 지점을 찾는 것이 중요하다.
* Bootstrapping
  * 학습데이터를 random으로 샘플링 하여 사용하는 것
  * 여러 묶음을 만들어 여러개의 모델을 학습하는 것에 사용
* Bagging and boosting
  * Bagging
    * 여러개의 모델을 sub 샘플링을 통해 학습
    * 여러개의 모델의 output을 voting, averaging을 이용해 사용
  * Boosting
    * 여러개의 모델을 Seqential하게 생성
    * 분류하기 어려운 데이터에 가중치 부여

### 1.3 Practical Gradien Decent Methods
* 구분
  * Stochastic GD
    * 한 single sample로 부터 업데이트
  * Mini-batch GD
    * Batch 크기만큼으로 부터 업데이트
  * Batch GD
    * 모두 사용해 업데이트
* Batch-size Matter
  ![](/assets/image/ML_AI/opt_3.png)
  큰 배치사이즈를 활용하면 sharp minimizers, 작은 배치 사이즈를 사용하면 Flat minimizers. Flat minimizers는 일반적으로 Test셋에 대해서도 잘 동작할 가능성이 높지만, sharp는 성능이 떨어 질 수도 있다.
* 종류
  ![](/assets/image/ML_AI/opt_4.png)
  출처:https://www.slideshare.net/yongho/ss-79607172
  * Stochastic Gradient Descent
    * $W_{t+1} \leftarrow W_t - r g_t$
    가장 큰 문제점? learning rate, step size를 결정하는 것이 어려움!
  * Momentum
    * $a_{t+1} \leftarrow  \beta a_t - g_t$
    * $W_{t+1} \leftarrow  W_t - ra_{t+1}$
    관성을 이용 이전 step의 gradient 방향을 일정 부분 유지. $\beta$ 가 momentum.
  * Nesterov Accelerate
    * $a_{t+1} \leftarrow  \beta a_t - \nabla L(W_t-r \beta a_t)$
    * $W_{t+1} \leftarrow  W_t - ra_{t+1}$ 
    $\nabla L(W_t-r \beta a_t)$ 는 Lookahead gradient라고 부른다. 한번 이동해보고, 그곳에서 계산을 진행해봄. 만약 Local minimum을 지났으면 momentum은 관성으로 인해 지나친 방향으로 더 가려는 습성을 지나는데,  Lookahead gradient을 이용하면 한번 더 간 곳의 gradient의 방향을 활용하기 때문에 조금 더 빨리 Local minimum에 수렴할 수 있다.
  * Adagrad
    * $W_{t+1} \leftarrow  W_t - {r \over \sqrt {G_t + \epsilon}}g_t$
    neural network 지금 까지 변해온 정보활용. 많이 변했던 것들은 적게, 적게 변했던 것들은 많이! 정보를 $G$ 에 저장 $\epsilon$ 0으로 나누는 것을 방지.
    하지만! $G$가 계속커지면 시간이 지나면 학습이 안되는 문제 발생
  * Adadelta
    * $G_t=rG_{t-1}+(1-r)g_t^2$
    * $W_{t+1} \leftarrow  W_t - {\sqrt {H_{t-1} + \epsilon} \over \sqrt {G_t + \epsilon}}g_t$
    * $H_t=rH_{t-1}+(1-r)(\Delta W_t)^2$
    Adagrad의 $G$가 계속해서 커지는 것을 방지하기 위해 나온 방법. Window size만큼의 gradient만 활용. 
    하지만 learning rate가 없어 잘 활용하지 않는다.
  * RMSprop
    * $G_t=rG_{t-1}+(1-r)g_t^2$
    * $W_{t+1} \leftarrow  W_t - {lr \over \sqrt {G_t + \epsilon}}g_t$
    힌턴이 이러면 잘되더라 했던 알고리즘. Stepsize를 분자에 추가한게 전부.
  * Adam
    * Momentum: $m_t=\beta_1 m_{t=1} + (1-\beta_1)g_t$
    * EMA of gradient squares: $v_t=\beta_2 v_{t-1} + (1-\beta_2)g_t^2$
    * $W_{t+1} = W_t + {lr \over \sqrt {v_t + \epsilon}} {\sqrt {1-\beta_2^t} \over 1-\beta_1^t}m_t$
    Momentum을 얼마나 유지할 것인가? EMA 정도, step size 등의 파라미터를 잘 설정해주는 것이 중요.

### 1.4 Regularization
Generalization이 잘 되게 하고 싶은것. 즉, Test 데이터에 대해서도 잘 동작할 수 있도록 규제하는 것이 목적
* Early Stopping
  * 검증 에러가 높아지는 시점에 학습 중단
* Parameter Norm Penalty
  * $total cost = loss(D;W)+{\alpha \over 2} \Vert W \Vert_2^2$
  네트워크 파라미터들의 크기를 줄이자. Function space에서 부드러운 함수로 보자.
* Data Augmentation
  많은 데이터는 언제나 좋은 결과를 가저다 준다. 
  이미지 같은 경우, 돌리거나, 뒤집거나 등의 방식을 통해 데이터를 늘릴 수 있음
  하지만 Mnist같은 경우에는 라벨이 변할 수 있으니(6->9) 주의할 필요가 있음
* Noise Robustness
  입력 데이터에 Noise를 중간중간 추가.
* Label Smooting
  학습데이터에서 임의적으로 선택한 두개의 데이터를 섞는것, 라벨과 이미지를 모두 섞음(Mix up, Cut mix)  
  ![](/assets/image/ML_AI/opt_5.png)
* Dropout
  일부 뉴런을 0으로 설정. forward pass
* [Batch Normalization](https://kyunghyunlim.github.io/ml_ai/2021/07/31/Batchnorm.html)
  * Batch norm
  * Layer norm
  * Instance norm
  * Group norm
  


  
