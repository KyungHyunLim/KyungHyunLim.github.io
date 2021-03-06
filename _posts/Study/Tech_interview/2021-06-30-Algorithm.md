---
layout: post
title:  "Algorithm (알고리즘)"
date:   2021-06-30 15:05:56 +0530
categories: [Tech_interview]
---
## 4.1 Basic
### 4.1.1 피보나치 수열 구현 방식 3가지, 시간복잡도 및 공간복잡도
```
- 재귀적 풀이: 시간복잡도 O(2^N), 공간복잡도: 동일(함수호출)
def fibo(n):
  return fibo(n-1) + fibo(n-1) if n >= 2 else n
 
- 반복문 풀이: 시간복잡도 O(N), 공간복잡도: constant
def fibo(n):
  if n < 2: 
    return n
  a, b = 0, 1
  for i in range(n-1):
    a, b = b, a + b
  return b
  
- 동적계획법을 이용한 풀이
def fibo(n):
  if n < 2:
    return n
  if dp[n] != 0:
    retrun dp[n]
  return dp[n] = fibo(n-1) + fibo(n-2)
```
### 4.1.2 빅오 표기법에 대해서 설명해주세요.
```
빅오 표기법은 알고리즘의 시간 복잡도를 나타냅니다. 알고리즘 내 연산의 횟수와 관련이 있습니다. 
예를 들어 반복문이 데이터의 수에 따라 제곱으로 늘어나면 시간복잡도는 N^2이 됩니다.
수학적으로는 어떤 알고리즘이 함수 G(n)이라면, 어떠한 n을 대입해도 c*g(n)이 되는 작은 상수 c가 존재함을 의미합니다.
```
### 4.1.3 팩토리얼을 구현해 보세요
```
def factorial(n)
  return n * factorial(n-1) if n != 1 else 1
```
### 4.1.4 BFS, DFS 차이는 무엇인가요?
```
노드를 방문하는 순서입니다. 
DFS는 깊이 우선 탐색으로 끝까지 탐색한 후에 돌아와 다른 방향을 탐색하는 방식이고, 
BFS는 너비 우선 탐색으로 현재 노드에서 갈 수 있는 노드들을 모두 확인하고 그 노드들 중 하나를 선택하는 방식으로 탐색합니다.
```
### 4.1.5 프림 알고리즘에 대해서 설명해 주세요.
```
그래프에서 노드 하나를 기준으로 삼아 연결된 다 노드로 갈 때 가장 적은 비용이 드는 엣지를 선택해 나아가는 알고리즘입니다. 
최소 신장 트리를 만들기 위한 방법 중 하나입니다.
```
### 4.1.6 다익스트라 알고리즘에 대해서 설명해 주세요.
```
유명한 최단 경로 탐색 알고리즘입니다. 
노드의 수^2만큼의 배열을 선언하고, 무한대 값을 할당합니다.
이제 출발 노드를 기준으로 각 노드까지의 최소 비용을 업데이트해 나아가면, 
최종적으로 배열에 행 노드에서 열 노드로 가는 최소 비용이 저장됩니다. 
```
### 4.1.7 은행원 알고리즘에 대해서 설명해 주세요.
```
데드락 상태를 회피하기 위한 알고리즘입니다.
안정상태와 불안정상태를 구분해 안정상태를 유지할 수 있을 때에만 자원을 할당해주는 알고리즘입니다. 
안정상태는 각 프로세스가 요구한 자원의 양이 소지한 자원의 양을 넘지 않도록 배정해 줄 수 있는 순서가 존재하는 상태를 말합니다. 
즉, 은행원 알고리즘을 적용하기 위해서는 각 프로세스들이 요구할 자원의 양, 
현재 프로세스가 사용중인 자원의 양, 가용한 자원의 양을 알 수 있어야 합니다.
```
## 4.2 정렬
### 4.2.1 정렬의 종류에는 어떤것들이 있나요?
```
버블, 선택, 삽입, 머지, 퀵, 힙 정렬 등이 있습니다.
```
### 4.2.2 삽입 정렬이 일어나는 과정을 설명해 보세요.
```
삽입 정렬은 자신보다 왼쪽에 작은 값이 나올 때까지 탐색 후 해당 위치에 삽입(스왑)하는 방식을 반복합니다. 
```
### 4.2.3 퀵 정렬이 일어나는 과정을 설명해 보세요.
```
좌측과 우측 값을 기준 값과 비교해, 교환 후 분할하는 정렬방식입니다.
좌측 값이 기준 값 보다 크거나 같을 때까지 비교합니다.
우측 값이 기준 값 보다 작거나 같을 때까지 비교합니다.
좌측 값과 우측 값을 교환합니다.
좌측 인덱스가 우측 인덱스 보다 크면 배열을 분할합니다.
재귀적으로 반복합니다.
```
### 4.2.4 54321 배열이 있을 때, 어떤 정렬을 사용하면 좋을까요?
```
이미 내림차순으로 정렬되어 있기 때문에, 사용 목적에 따라 인덱스를 역순으로 접근하면 가장 시간적 효율이 높다고 생각합니다. 
그럼에도 정렬을 해야 한다면, 가장 빠른 시간복잡도로 알려진 힙 정렬이나 병합 정렬을 사용할 것 같습니다.
```
### 4.2.5 랜덤으로 배치된 배열이 있을때, 어떤 정렬을 사용하면 좋을까요?
```
배열의 상태에 구애받지 않고 일정한 시간 복잡도를 가진 힙 정렬이나 병합 정렬을 사용할 것 같습니다.
```
### 4.2.6 자릿수가 모두 같은 수가 담긴 배열이 있을 때, 어떤 정렬을 사용하면 좋을까요?
```
공간 복잡도가 높아지기는 하지만, 기수 정렬을 사용하는 것이 시간적으로 효율이 가장 높다고 생각합니다. 
기수 정렬은 일의 자리부터 순서대로 자릿수 별로 버킷에 담는 방식으로,
최대 자릿수 만큼만 반복하면 되기 떄문에 빠르게 정렬이 가능합니다.
```
