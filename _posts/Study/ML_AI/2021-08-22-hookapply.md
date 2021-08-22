---
layout: post
title:  "Hook and Apply"
date:   2021-08-22 15:52:12
categories: [Pytorch, ML_AI]
use_math: true
---

## 1. Intro
 apply는 많이 쓰이지만 hook은 사실 잘 쓰이는 기능은 아니라고 한다. 하지만 알아두면 언젠가 유용할 것 같다. hook을 이용해서 gradient, forward 계산값등 다양한 조절이 가능하고, 그걸 apply를 통해 원하는 모듈에만 적용하게도 만들 수 있다. 최근 pretrained 모델을 많이 사용하는데, 새로운 시도를 적용해보기 적절한 기능들인것 같다.

## 2. hook
    * 프로그램의 실행 로직을 분석하거나
    * 프로그램에 추가적인 기능을 제공하고 싶을 때 사용
    * Forward hook

    ```python
    # TODO: 답을 x1, x2, output 순서로 list에 차례차례 넣으세요! 
    answer = []

    # TODO : pre_hook를 이용해서 x1, x2 값을 알아내 answer에 저장하세요
    def pre_hook(module, input):
        for v in input:
            answer.append(v)
        pass
    # TODO : hook를 이용해서 output 값을 알아내 answer에 저장하세요
    def hook(module, input, output):
        answer.append(output)
        pass
    add.register_forward_pre_hook(pre_hook)
    add.register_forward_hook(hook)
    --------------------------------------------------------------------
    # TODO : hook를 이용해서 전파되는 output 값에 5를 더해보세요!
    def hook(module, input, output):
        output = output + 5
        return output
        pass

    add.register_forward_hook(hook)
    ```

    * Backward hook

    ```python
    # TODO: 답을 x1.grad, x2.grad, output.grad 순서로 list에 차례차례 넣으세요! 
    answer = []

    # TODO : hook를 이용해서 x1.grad, x2.grad, output.grad 값을 알아내 answer에 저장하세요
    def module_hook(module, grad_input, grad_output):
        for gi in grad_input:
            answer.append(gi)
        answer.append(grad_output[0])
        pass

    model.register_full_backward_hook(module_hook)
    ---------------------------------------------------------------
        
    # TODO : hook를 이용해서 W의 gradient 값을 알아내 answer에 저장하세요
    def tensor_hook(grad):
    #     print(grad)
        answer.append(grad)
        pass

    model.W.register_hook(tensor_hook)
    ```
## 3. apply
    * apply 함수는 일반적으로 가중치 초기화(Weight Initialization)에 많이 사용
    * apply를 통해 적용하는 함수는 모든 module들을 순차적으로 입력받아서 처리

    ```python
    # TODO : apply를 이용해 모든 Parameter 값을 1로 만들어보세요!
    def weight_initialization(module):
        module_name = module.__class__.__name__
        if module_name.split('_')[0] == 'Function':
            module.W = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)
            #print(type(module))

    # 🦆 apply는 apply가 적용된 module을 return 해줘요!
    returned_module = model.apply(weight_initialization)
    ```
    * 기존 모델에 파라미터 추가하고, forward 방식 바꾸기

    ```python
    # TODO : apply를 이용해 Parameter b를 추가해보세요!
    def add_bias(module):
        module_name = module.__class__.__name__
        # 각 Function 모듈에 bias 추가
        if module_name.split('_')[0] == "Function":
            module.b = Parameter(torch.rand(2, ), requires_grad=True)
            

    # TODO : apply를 이용해 추가된 b도 값을 1로 초기화해주세요!
    def weight_initialization(module):
        module_name = module.__class__.__name__

        if module_name.split('_')[0] == "Function":
            module.W.data.fill_(1.)
            # Function 모듈 bias 초기화
            module.b.data.fill_(1.)


    # TODO : apply를 이용해 모든 Function을 linear transformation으로 바꿔보세요!
    #        X @ W + b
    def linear_transformation(module):
        module_name = module.__class__.__name__
        
        # hook을 이용해 forward 함수 출력값 선형으로 변형
        def hook(moudle, input, output):
            output = (torch.matmul(input[0], module.W.T) + module.b)
            return output
        
        # 각 모듈에 hook 등록
        if module_name == "Function_A":
            module.register_forward_hook(hook)
        elif module_name == "Function_B":
            module.register_forward_hook(hook)
        elif module_name == "Function_C":
            module.register_forward_hook(hook)
        elif module_name == "Function_D":
            module.register_forward_hook(hook)
            
    returned_module = model.apply(add_bias)
    returned_module = model.apply(weight_initialization)
    returned_module = model.apply(linear_transformation)

    ```