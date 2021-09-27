---
layout: post
title:  "자연어의 전처리"
date:   2021-09-27 13:45:22
categories: [NLP, ML_AI]
use_math: true
---

# 1. 자연어 전처리
## 1.1 자연어처리의 단계
* 자연어 전처리?
    * Raw data를 기계 학습 모델이 학습하는데 적합하게 만드는 프로세스
    * 학습에 사용된 데이터를 수집&가공하는 모든 프로세스
* 왜 필요한가?
    * 가장 중요한 것은 데이터이기 때문
    * Task의 성능을 가장 확실하게 올릴 수 있는 방법!
* 단계
    1. Task 설계
        * 요구사항 분석 
        * E.g. 유튜브 라이브에서 악성 댓글 필터링 해주세요! $\rightarrow$ Task: 악성 댓글 분류 
    1. 필요 데이터 수집
    1. 통계학적 분석
        * Token 개수 $\rightarrow$ outlier 제거
        * 빈도 확인 $\rightarrow$ 사전 정의
    1. 전처리
        * 개행문자 제거, 특수문자 제거, 공백 제거, 중복 표현 제거, 이메일/링크 제거, 제목 제거, 불용어 제거, 조사 제거, 띄어쓰기/문장분리 보정
    1. Tagging (데이터 라벨링)
    1. Tokenizing
        * 어절
        * 형태소
        * WordPiece
    1. 모델 설계
    1. 모델 구현
    1. 성능 평가
    1. 완료

## 1.2 Python string 관련 함수
* 대소문자 변환
    * upper()
    * lower()
    * capitalize()
    * titile()
    * swapcase()
* 편집, 치환
    * strip()
    * rstrip()
    * lstrip()
    * replace(a, b)
* 분리, 결합
    * split()
    * ''.join(list)
    * lines.splitlines()
* 구성 문자열 판별
    * isdigit()
    * isalpha()
    * isalnum()
    * islower()
    * isupper()
    * isspace()
    * startswith(string)
    * endswith(string)
* 검색
    * count('string')
    * find('string')
    * find('string', 3)
    * rfind('string')
    * index('string')
    * rindex('string')

## 1.3 유용한 한국어 라이브러리
* kss: 문장 분리
* soynlp: 반복되는 글자 줄이기
* git+https://github.com/haven-jeon/PyKoSpacing.git: 띄어쓰기 교정
* git+https://github.com/ssut/py-hanspell.git: 맞춤법
* konlpy: 형태소 분석

# 2. 자연어 토크나이징
## 2.1 한국어 토큰화
* 토큰화(Tokenizing)
    * 주어진 데이터를 Token 단위로 나누는 작업
    * Token이 되는 기준은 상이함 (어절, 단어, 형태소, 음절, 자소 등)
    * 문장 토큰화: 문장 분리
    * 단어 토큰화: 구두점 분리, 단어 분리
* 한국어 토큰화
    * 영어는 New York나 it's 같은 합성어나 줄임말을 예외처리하면 띄어쓰기를 기준으로도 잘 작동하는 편
    * 한국어는 조사나 어미를 붙어서 말을 만드는 언어로, 띄어쓰기만으로는 어려움
        * E.g 그, 그가, 그는, 그를, 그에게
    * 어절이 의미를 가지는 최소 단위인 형태소로 분리
        * E.g. 안녕하세요 $\rightarrow$ 안녕, 하, 세, 요

# 3. Further Reading
* [추가자료(소개)](https://www.youtube.com/watch?v=9QW7QL8fvv0)
* [추가자료(실습)](https://www.youtube.com/watch?v=HIcXyyzefYQ)

# 4. Further Question
* 텍스트 정제라는 것이 정말 필요할까요?
    * 어쩌라는거야? 싶으시죠? ☺️☺️
    * 실제로 우리가 웹이나 메신저를 통해 사용하는 언어는 '정제 되지 않은 언어' 입니다.
    * 해당 데이터가 적용되는 방향에 따라 정제가 필요할 수도, 필요하지 않을 수도 있습니다.
    * 오히려 더욱 어려운 데이터로 학습한 모델의 성능이 좋을 수도 있죠 ☺️
 