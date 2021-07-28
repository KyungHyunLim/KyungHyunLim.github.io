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
Type I error는 false positive인 반면에 Type II error는 false negative를 의미합니다. 다시말하자면,
Type I error는 어떤 일이 일어나지 않았을 때 어떤 일이 일어났다고 주장하는 것이고 (false -> true),
Type II error는 어떤 일이 실제로 일어났을 때, 아무 일도 일어낙지 않는다고 주장한 것을 의미합니다. (true -> false)
```
### 11.1.11 What’s a Fourier transform?
```
퓨리에 트랜스폼은 X라는 도메인에서 다른 도메인으로 신호의 특성을 변환하는 방법을 의미합니다. 주로 시계열 데이터의 특성을 찾기 위해 사용되며, 특히 음성 신호와 같은 데이터에서 시간 영역에서 주파수 영역으로 데이터를 변환해 특성을 활용하기 위해 사용합니다.
```
### 11.1.12 What’s the difference between probability and likelihood?
```
Probability는 possible result에 기초하고,
likelihood는 hypothesis를 기반으로 합니다.

간단하게 말하자면,
probability는 관찰 결과에 의해 계산되는 값입니다.
likelihood는 어떤 분포를 추정해, 그 분포에서 관찰될 사건이 일어날 확률. 다시말해 어떤 분포에서 그 사건이 관찰될 확률을 의미합니다.

+a
probability는 multually exclusive 하고
likelihood는 multually exclusive하지 않다.

상호 베타적이라는 것은 P(A or B) = P(A) + P(B)를 만족한다.
* 독립은 P(A or B) = P(A) + P(B) - P(A)P(B)를 만족한다. 
```
### 11.1.13 What is deep learning, and how does it contrast with other machine learning algorithms?
```
딥러닝은 신경망으로 구성된 기계 학습의 한 종류입니다. 즉, 딥러닝은 역전파 알고리즘과 신경망 구조를 특정 원칙을 사용해 레이블이 지정되지 않거나 반구조화된 큰 데이터의 집합을 보다 정확하게 모델링하는 방법입니다.
```
### 11.1.14 What’s the difference between a generative and discriminative model?
```
생성 모델은 데이터 범주를 학습하는 반면 판별 모델은 단순히 데이터의 다른 범주 차이를 학습합니다. 
판별 모델은 일반적으로 분류 작업에서 생성 모델보다 성능이 좋습니다.
```
### 11.1.15 What cross-validation technique would you use on a time series dataset?
```
cross-validation 모델이 정말 원하는 방향으로 학습된 것인지, 단순히 현재 데이터에 피팅된 것인지 파악할 수 있는 방법입니다.
훈련데이터와 평가데이터의 구성을 다르게해 여러차례 학습과 평가를 반복해 일정한 성능을 보여주는지 확인하는 것이 목적입니다.

시계열 데이터에 적용할 경우
시계열 데이터의 특성을 이용해
과거와 예측할 미래를 나누는 시점을 점점 미래로 옮겨가며 구성할 수 있습니다.
```
### 11.1.16 How is a decision tree pruned?
```
프루닝이란 모델의 복잡성을 줄이고 의사결정 트리 모델의 예측 정확도를 높이기 위해 예측 능력이 약한 가지를 제거하는 것입니다. 
프루닝은 오류 와 비용, 복잡성 제거와 같은 접근 방식을 통해 상향식 및 하향식 두가지 방식으로 수행될 수 있습니다.

가장 간단하게 예측정확도가 줄어들지 않으면 노드들을 잘라내는 방식을 활용할 수 있습니다.
```
### 11.1.17 Which is more important to you: model accuracy or model performance?
```
모델의 성능이 더 중요합니다.
만약 이상탐지 모델을 만든다고 할 때, 일반적으로 대부분의 데이터는 정상데이터로 이루어져있습니다.
하지만, 이상탐지 모델은 소수의 비정상 데이터를 분류해내야합니다.
이때, 정확도만으로 모델을 판단한다면, 이 모델은 고성능의 모델로 판단되겠지만
실질적으로 아무런 쓸모없는 모델일 수 있기 때문입니다.
따라서, 정확도만을 중요한 메트릭으로 사용하는 것은 한계가 있을 수 있습니다.
```
### 11.1.18 What’s the F1 score? How would you use it?
```
F1 score = 2 * (Precision * Recall) / (Precision + Recall)
=> 데이터가 불균형 구조일 때, 모델의 성능을 정확하게 평가 가능 (조화평균)

둘중하나가 0과 가까우면 낮은 수치를 보여주기 때문에, 보다 정확하게 모델을 평가할 수 있습니다.
```
### 11.1.19 How would you handle an imbalanced dataset?
```
첫번째로는, 더 많은 데이터를 수집하는 방법이 있습니다.
두번째로는, augmentation 기법을 활용해 데이터를 늘리는 방법입니다.
마지막으로는, 데이터를 다시 샘플링해 불균형을 완화하는 방법입니다.

중요한 것은 불균형으로 인해, 모델이 편향된 학습을 하지 않도록 주의하는 것이라고 생각합니다.
```
### 11.1.20 When should you use classification over regression?
```
분류 문제는 범주형 데이터를 , 회귀 문제는 연속형 값을 결과로 제공합니다..
한 입력에 대해 어떤 범주인지를 나누는 것이 목적일때, 분류를 사용합니다.
```
### 11.1.21 Name an example where ensemble techniques might be usful.
```
ensemble technique은 서로 다른 알고리즘들의 예측을 voting 또는 평균을 linear combination해 더 좋은 예측을 만들기 위한 것입니다.
오버피팅을 방지할 수 있고, 더 강건한 모델을 만들 수 있습니다.

앙상블을 위한 기술들에는 bagging, boosting 등이 있습니다.

+a
bagging(bootstrap aggregating):
    1. 복원추출을 통해 n개의 샘플 모음 생성
    2. 해당 샘플에 대해 모델 학습
    3. 1-2 를 일정 횟수 반복
boosting: 
    1. weak learner를 생성후 error 계산
    2. Error에 기여한 샘플들에 다른 가중치를 주어 새로운 모델 학습
    3. 1-2 반복
```
### 11.1.22 How do you ensure you're not overfitting with a model?
```
1. 모델을 너무 복잡하지 않도록 설계
    * 변수와 모수가 적어짐
2. K-fold와 같은 교차 검증 사용
3. LASSO와 같은 정규화 기법을 사용 (L1 norm)
```
### 11.1.23 What evaluation approaches would you work to gauge the effectiveness of a machine learning model?
```
데이터셋을 훈련, 검증, 평가 3가지로 분할합니다.
또한, 교차 검증을 활용해 학습된 모델을 평가합니다.
이때, 적절한 메트릭을 선택하는 것 또한 중요합니다.
```
### 11.1.24 How would you evaluate a logistic regression model?
```
logistic regression 모델의 목적에 따라 적절한 평가방법을 선택해야 합니다.

예를들어, 
어떤 환자의 정보를 가지고 암에 걸렸는지, 걸리지 않았는지를
판단하는 모델이라면, 정확도, 정밀도, 재현도, f1-score 등을 활용해 평가를 할 수 있습니다.
또 다른 경우로,
어떤 환자의 정보를 바탕으로 간수치를 예측한다면, MSE, MAE와 같은 평가지표를 활용해 얼마나 근접하게 예측했는지 평가할 수 있습니다.
```
### 11.1.25 What’s the “kernel trick” and how is it useful?
```
커널 트릭은 커널 함수를 이용해, 저차원의 데이터를 고차원 공간에 표현해 주는 것을 의미합니다.
저차원에서는 구분할 수 없었던 데이터들이 고차원에 표현되면서
보다 많은 특징이 들어나고 효과적으로 분리하기 위한 알고리즘들을 실행할 수 있습니다. 
```
## 11.2 Programming
### 11.2.1 How do you handle missing or corrupted data in a dataset?
```
Panda에서는 isnull()과 dropna()의 두 가지 매우 유용한 방법이 있습니다.
isnull()과 dropna()는 결측 데이터 또는 손상된 데이터가 있는 데이터 열을 찾아 해당 값을 삭제하는 데 도움이 됩니다.
잘못된 값 채우려면 fillna() 방법을 사용할 수 있습니다.
```
### 11.2.2 Do you have experience with Spark or big data tools for machine learning?
```
mysql ?
hp5y
HDF5matrix
```
### 11.2.3 Pick an algorithm. Write the pseudo-code for a parallel implementation.
```
https://stackoverflow.com/questions/5583257/writing-pseudocode-for-parallel-programming
```
### 11.2.4 What are some differences between a linked list and an array?
```
배열은 데이터를 입력하면 순차적으로 입력이 되며, 물리적 주소 또한 순차적입니다. 
또한 한번 크기가 결정이 되면 변경이 불가능 합니다. 하지만 인덱스가 있어 데이터에 접근하는 속도가 빠릅니다. 
하지만 중간에 삽입하거나 삭제하는 경우 인덱스를 재배열해야 하기 때문에 효율이 떨어집니다.

링크드 리스트는 다음 원소의 주소 값을 가지고 있어, 이를 이용해 리스트를 생성합니다. 
따라서 크기가 가변적이고, 삽입과 삭제가 주소 값을 변경해 주면 되기 때문에 간편합니다.
하지만 중간에 있는 원소에 접근하기 위해서는 포인터를 따라 탐색이 필요해, 배열과 같이 한번에 접근할 수가 없습니다.
```
### 11.2.5 Describe a hash table.
```
해시 테이블은 연관된 배열을 생성하는 데이터 구조입니다.
해시 함수를 이용한 어떤 값과 키가 매칭이됩니다.
보통 데이터베이스에서 인덱싱을 위해 사용합니다.
```
### 11.2.6 Which data visualization libraries do you use? What are your thoughts on the best data visualization tools?
```
matplot.pyplot 또는 seaborn 라이브러리를 사용합니다.
다양한 방법으로 데이터를 시각화할 수 있는 기능을 제공하기 때문에 두 라이브러리를 적절이 사용한다면 대부분 원하는 방법으로 시각화가 가능하다고 생각합니다.
```
### 11.2.7 Given two strings, A and B, of the same length n, find whether it is possible to cut both strings at a common point such that the first part of A and the second part of B form a palindrome.
```
팰린드롬을 확인하기 위한 방법은 여러가지가 있지만,
가장 간단한 방법으로, 
A와 B의 가능한 부분 문자열들을 모두 구하고,
두 서브그룹간 팰린드롬이 형성될 수 있는지 검사하는 방법을 사용할 것 같습니다.
```