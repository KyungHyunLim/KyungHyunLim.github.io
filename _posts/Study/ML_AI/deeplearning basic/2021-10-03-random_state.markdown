---
layout: post
title:  "Scikit-learn train_test_split의 random_state"
date:   2021-10-03 15:56:22
categories: [ML_AI]
use_math: true
---

# 1. Random_state는 정말 재현 가능성을 보장 해줄까?
* 실험 계획 - 다른 환경에서 돌려보자!
    * 외부 서버 - jupyter notebook, Console
    * 노트북 로컬 - jupyter notebook
    * 구글 colab
* 데이터 - 간단한 array

```python
test_list = [1,2,3,4,5,6,7,8,9,10]
print(train_test_split(test_list, randome_state=42))
```

# 2. 결과
* 외부 서버  
    ![](/assets/image/ML_AI/2.PNG)  
    ![](/assets/image/ML_AI/1.PNG)
* 노트북 로컬  
    ![](/assets/image/ML_AI/4.PNG)
* 구글 코랩  
    ![](/assets/image/ML_AI/3.PNG)

# 3. 결론
* 외부 서버, 노트북 로컬, 구글 코랩은 완전히 서로 다른 환경이다. 하지만 비록 9개의 작은 샘플이긴 하지만 모두 같은 결과를 보여주었다. 따라서 재현가능성을 보장해준다!