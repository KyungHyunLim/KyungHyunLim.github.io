---
layout: post
title:  "Codility LongestPassword (Python)"
date:   2021-11-12 21:22:12
categories: [Algorithm]
use_math: True
---

# 1. 문제 설명
* [문제 링크](https://app.codility.com/programmers/trainings/1/longest_password/)  
주어진 문자열에 대해서, 각 단어들을 password 후보로 생각했을 때, 후보들 중 아래 조건에 맞는 가장 긴 길이는?
1. password는 알파벳 소문자, 대문자, 0-9 사이 숫자로만 이루어져야한다.
2. password에서 알파벳은 짝수개가 존재해야 한다.
3. password에서 숫자는 홀수개가 존재해야 한다.
가능한 후보가 없을 경우 -1 출력

# 2. 풀이
* 알고리즘 문제라기 보다는 간단한 구현문제이다. 주어진 문자열을 띄어쓰기 단위로 split하고, 각 단어에 대해서 1,2,3 조건을 검사한다.

# 3. 코드
* python

```python
import re

def alpha_num_check(word):
    cp = re.compile(r'[^a-zA-Z0-9]+')
    isok = cp.findall(word)
    if isok:
        return False
    return True

def even_alpha_check(word):
    cp = re.compile(r'[a-zA-Z]+')
    isok = cp.findall(word)
    if len(''.join(isok)) % 2 == 1:
        return False
    return True

def odd_num_check(word):
    cp = re.compile(r'[0-9]+')
    isok = cp.findall(word)
    if len(''.join(isok)) % 2 == 0:
        return False
    return True

def solution(S):
    answer = -1
    for word in S.split():
        if alpha_num_check(word) and even_alpha_check(word) and odd_num_check(word):
            if answer < len(word):
                answer = len(word)
    return answer
```

# 4. 결과
![](/assets/image/Algorithms/cd_lp_1.PNG)  
간단하게 풀수 있는 문제였다. 내일 보는 코딩테스트 플렛폼이 코딜리티여서 경험삼아 풀어보았다.