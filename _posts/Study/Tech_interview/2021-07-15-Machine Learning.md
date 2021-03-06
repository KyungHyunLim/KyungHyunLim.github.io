---
layout: post
title:  "Machine Learning"
date:   2021-07-15 19:54:21 +0530
categories: [Tech_interview]
---
## 11.1 BASIC
### 11.1.1 What’s the trade-off between bias and variance?
```
Bias는 학습에 사용 중인 알고리즘의 잘못된 가정이나, 단순한 가정으로 인한 오류 입니다.
Variance는 사용 중인 알고리즘이 너무 복잡해 발생하는 오류 입니다.
따라서 두 변수는 반비례 관계에 있으며, 학습을 통해 둘 모두를 최소화 할수 있는 지점을 찾는 것이 목적이 됩니다.
```
### 11.1.2 What is the difference between supervised and unsupervised machine learning?
```
supervised는 지도학습으로, 정답지를 가지고 있는 데이터를 사용해 학습합니다.
입력과 출력을 짝지어 학습을 시키는 것이 가능합니다.

unsupervised는 비지도학습으로, 정답지가 없는 경우에 비지도 학습을 주로 사용합니다.
대표적으로 강화학습이 있으며, 입력과 현재 상태에 대해 스코어같은 것을 통해 보상이나 처벌을 주어
목표를 향해 행동할 수 있도록 학습을 진행합니다.
```
### 11.1.3 How is KNN different from k-means clustering?
```
KNN(K-Nearest Neighbors)는 지도학습 기반의 분류 알고리즘 입니다.
k-means clustering은 비지도 학습 기반의 클러스터링을 위한 알고리즘 입니다.

KNN은 새로운 데이터에 대해 가까운 주변 데이터의 라벨이 필요하고,
k-means clustering은 일정 군집 형성 후 threshold를 정해 그 범위를 가지고 어떤 군집에 속하는지
결정하게 됩니다.
```
### 11.1.4 Explain how a ROC curve works.
```
ROC커브는 True-positive rate와 False-positive rate 사이 에서 treshold를 변경해가며 측정한것을 
시각화한 그래프입니다.
주로, Sensitivity(True-positive)와 오류(False-positive)를 유발할 확률 사이의 절충점을 결정, 참고하기 위해
사용됩니다.

+a
True positive(TP) = 실제 True인 정답을 True라고 예측한 수 - 정답
False Positive(FP) = 실제 False인 정답을 True라고 예측한 수 - 오답
False Negative(FN) = 실제 True인 정답을 False라고 예측한 수 - 오답
True Negative(TN) = 실제 False인 정답을 False라고 예측한수 - 정답

Precision = TP / (TP + FP)
=> True라고 분류한 것중 실제 True 비율

Recall = TP / (TP + FN)
=> 실제 True인 것중 모델이 True라고 분류한 비율
```
### 11.1.5 Define precision and recall.
```
Precision = TP / (TP + FP)
=> True라고 분류한 것중 실제 True 비율

Recall(Sensitivity) = TP / (TP + FN)
=> 실제 True인 것중 모델이 True라고 분류한 비율

+a
F1 score = 2 * (Precision * Recall) / (Precision + Recall)
=> 데이터가 불균형 구조일 때, 모델의 성능을 정확하게 평가 가능 (조화평균)
```
### 11.1.6 What is Bayes’ Theorem? How is it useful in a machine learning context?
```
Bayes' Theorem은 사전 지식으로 알려진 사건의 사후 확률을 제공합니다.

수학적으로, Sample의 positive rate를 모집단의 FP rate와 조건의 positive rate의 합으로
나눈 값으로 표현됩니다.

True Positive Rate of a Condition Sample / 
True Positive Rate of a Condition Sample + False Positive Rate of a Population

```
### 11.1.7 Why is “Naive” Bayes naive?
```
모델이 추론함에 있어 각각의 변수들을 독립적인 것으로 가정하기 때문입니다.
이는 실제 세계에서 불가능한 일이기 때문에, 결코 충족될 수 없는 조건입니다.
따라서, "Naive"라는 키워드가 붙은 것으로 알고 있습니다.
```
### 11.1.8 Explain the difference between L1 and L2 regularization.
```
L2 regularization의 경우 오차에 제곱을 하기 때문에 L1에 비해 Outlier에 더 민감합니다.
따라서 두 정규화의 차이는,
L2는 전체적으로 오차를 줄이고 각각의 벡터에 대해 Unique한 값을 내지만,
L1의 경우 상황에 따라, 특정 Feature를 selection 하는 것이 가능하다는 차이가 있습니다.

+a
L2 Regularization 을 사용하는 Regression model 을 Ridge Regression 이라고 부릅니다.
L1 Regularization 을 사용하는 Regression model 을 Least Absolute Shrinkage and Selection Operater(Lasso) Regression 이라고 부릅니다.
```
### 11.1.9 What’s your favorite algorithm, and can you explain it to me in less than a minute?
```
딥 러닝은 신경망과 관련된 기계 학습의 하위 집합입니다. 
즉, 역전파 및 신경과학의 특정 원칙을 사용하여 레이블이 지정되지 않은 또는 반구조화된 데이터의 큰 집합을 보다 정확하게 모델링하는 방법입니다. 
그런 의미에서 딥 러닝은 신경망을 사용하여 데이터 표현을 학습하는 비지도 학습 알고리즘을 나타냅니다.
```
### 11.1.10 What’s the difference between Type I and Type II error?
```
생성 모델은 데이터 범주를 학습하는 반면 판별 모델은 단순히 데이터의 다른 범주 차이를 학습합니다. 
판별 모델은 일반적으로 분류 작업에서 생성 모델보다 성능이 좋습니다.
```
### 11.1.11 What’s a Fourier transform?
```

```
### 11.1.12 What’s the difference between probability and likelihood?
```

```
### 11.1.13 What is deep learning, and how does it contrast with other machine learning algorithms?
```

```
### 11.1.14 What’s the difference between a generative and discriminative model?
```

```
### 11.1.15 What cross-validation technique would you use on a time series dataset?
```

```
