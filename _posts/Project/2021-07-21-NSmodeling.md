---
layout: post
title:  "Noise surppression 모델-(pytorch)"
date:   2021-07-21 17:05:56 -5:30
categories: [Noise_surpression]
---
## 0. 모델 개요
 GAN 기반 잡음 제거 모델을 사용했습니다.
 End-to-End 잡음 제거 모델 중 에서는 가장 좋은 성능을 보여주고 있다고 생각합니다.

 Generator(G): 잡음 제거 담당
 Clean Discriminator(CD): G가 생성한 음성 신호와 타겟 음성 신호 비교
 Noise Discriminator(ND): 입력과의 차이를 이용해 Noise 파형을 얻어, 학습에 사용
 
## 1. 라이브러리 임포트
```python
import torch
from torch import nn
import torch.utils as utills
import numpy as np
import torchsummary
import torch.nn.functional as F
from torchgan.layers import VirtualBatchNorm
```

## 2. 판별자 구현
 CD와 ND는 유사한 구조로 이루어져 있습니다. 다만, 입력이 달라 다른 것을 학습하게 됩니다.
 정확히는 ND는 G의 인코더와 더 유사하다고 볼 수 있습니다.

 CD: (타겟 음성, 잡음 음성) -> true label
     (생성된 음성, 잡음 음성) -> fake label

 ND: (잡음 음성 - 타겟 음성) -> true label
     (잡음 음성 - 생성된 음성) -> fake label

```python
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.layers = nn.ModuleList()
        self.filters = [1, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
        
        for i in range(10):
            self.layers.append(nn.Sequential(
                nn.Conv1d(
                    in_channels=self.filters[i] * 2, 
                    out_channels=self.filters[i+1] * 2,
                    kernel_size = 32,
                    stride=2,
                    padding=15),
                VirtualBatchNorm(self.filters[i+1] * 2),
                nn.LeakyReLU(0.3)
                 )
             )
                              
        self.flatten = nn.Sequential(
            nn.Conv1d(1024, 1, kernel_size=1, stride=1),
            nn.LeakyReLU(0.3),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
                              
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.flatten(x)
        return x
```

## 3. 생성자 구현
 일반적으로 GAN은 랜덤 분포인 z를 생성해 입력으로 사용하는데, Speech enhancement GAN에서는 이 z를 생략하는 것이 성능이 더 좋게 나타난다는 연구가 있습니다.
 즉, Speech enhancement GAN의 생성자는 인코더에서 추출한 잠재공간 벡터를 바탕으로 잡음이 제거된 음성을 생성하도록 학습됩니다.
 또한 인코더의 출력을 디코더의 레이어에서 concat 함으로써 residual 구조를 사용하였습니다. 컨볼루션 층을 통과하며 손실되는 정보를 넘겨주어 보다 원본에 가까운 음성을 생성할 수 있는 효과가 있습니다.

 구현한 생성자 또한 forward 함수를 2개 가지고 있습니다. z를 사용할 때와 사용하지 않을 때 입니다.

```python
class NoiseCanceler(nn.Module):
    def __init__(self, skip_z):
        super(NoiseCanceler, self).__init__()
            
        self.enc_layers = nn.ModuleList()
        self.dec_layers = nn.ModuleList()
        self.filters = [1, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
        self.skip_z = skip_z
        
        self.prelu = nn.PReLU()
        
        # For Encoder [Batch x feature map x length]
        for i in range(11):
            self.enc_layers.append(nn.Conv1d(
                    in_channels=self.filters[i], 
                    out_channels=self.filters[i+1],
                    kernel_size = 32,
                    stride=2,
                    padding=15)
               )
               # output: [Batch x 1024 x 8]
       
        # For Decoder
        # Gaussian random variable z, Whether or not to use.
        for i in range(11, 0, -1):
            if i == 11 and skip_z == True:
                 self.dec_layers.append(nn.ConvTranspose1d(
                        in_channels=self.filters[i], 
                        out_channels=self.filters[i-1],
                        kernel_size = 32,
                        stride=2,
                        padding=15)
                    )
                  # output: [Batch x 1 x 16384]
            else:
                self.dec_layers.append(nn.ConvTranspose1d(
                        in_channels=self.filters[i] * 2, 
                        out_channels=self.filters[i-1],
                        kernel_size = 32,
                        stride=2,
                        padding=15)
                    )
                   # output: [Batch x 1 x 16384]  
        self.dec_tanh = nn.Tanh()

    # Use z      
    def forward(self, x, z):
        values = []
       
        # Encoding
        for enc in self.enc_layers:
            x = self.prelu(enc(x))
            values.append(x)
        
        # Enc out : Batch x 1024 x 8
        values.reverse()
        x = torch.cat((x, z), dim=1)
            
        # Decoding
        for idx, dec in enumerate(self.dec_layers):
            x = dec(x)
            if idx < 10:
                x = torch.cat((x, values[idx + 1]), dim=1)          
                        
        x = self.dec_tanh(x)
        return x
    
    # Not use z
    def forward(self, x):
        values = []
       
        # Encoding
        for enc in self.enc_layers:
            x = self.prelu(enc(x))
            values.append(x)
        
        # Enc out : Batch x 1024 x 8
        values.reverse()
        
        # Decoding
        for idx, dec in enumerate(self.dec_layers):
            x = self.prelu(dec(x))
            if idx < 10:
                x = torch.cat((x, values[idx + 1]), dim=1)          
                        
        x = self.dec_tanh(x)
        return x
```

## 4. Reference
```
1. Lim, K. H., Kim, J. Y., & Cho, S. B. (2019, November). Non-stationary noise cancellation using deep autoencoder based on adversarial learning. In International Conference on Intelligent Data Engineering and Automated Learning (pp. 367-374). Springer, Cham.
2. Lim, K. H., Kim, J. Y., & Cho, S. B. (2020, November). Generative Adversarial Network with Guided Generator for Non-stationary Noise Cancelation. In International Conference on Hybrid Artificial Intelligence Systems (pp. 3-12). Springer, Cham.
3. Pascual, S., Bonafonte, A., & Serra, J. (2017). SEGAN: Speech enhancement generative adversarial network. arXiv preprint arXiv:1703.09452.
```