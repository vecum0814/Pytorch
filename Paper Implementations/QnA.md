## transforms.toTensor 사용이유?

transforms.ToTensor(), # torchvision의 PIL 라이브러리를 이용하여 이미지를 읽을 때 (H, W, C), [0, 255]로 읽어오는데, ToTensor()로 (C, H, W), [0.0, 1.0]으로 변환


## 1 x 1 Convolution이란?


## Gradient Vanishing / Exploding에 대해서 설명해주세요

gradient vanishing이란 layer가 깊어질수록 미분을 점점 많이 하기 때문에 backpropagation을 해도

앞의 layer일수록 미분값이 작아져 그만큼 output에 영향을 끼치는 weight 정도가 작아지는 것을 말한다.

이는 overfitting과는 다른 문제인데 overfitting은 학습 데이터에 완벽하게 fitting시킨 탓에 테스트 성능에서는

안 좋은 결과를 보임을 뜻하고 위와 같은 문제는 Degradation 문제로 training data에도 학습이 되지 않음을 뜻한다.

## L1 Norm과 L2 Norm이란?

## Batch Normalization이란??


## Degradation Problem이란?
With the network depth increasing, accuracy gets saturated and then degrades rapidly.

## Latent Vector란? Latent Space는?

## 파이썬에서 super 함수는 어떤 기능을 할까?

## Upsampling과 Downsampling 각각의 기능과 사용하는 목적은?

## Transposed Convolution이란??
<img width="1057" alt="스크린샷 2022-08-11 오전 1 28 58" src="https://user-images.githubusercontent.com/52812351/183963588-628bf7be-f806-4a84-9005-0a272b650ada.png">

