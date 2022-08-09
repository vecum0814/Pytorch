ResNet implementation with with CIFAR-10 dataset
====================================================


## ResNet은 Skip Connection을 활용한 Residual Learning을 기반으로 보다 깊은 Layer를 쌓을 수 있는데 큰 공헌을 했습니다.

## ResNet의 탄생 배경
2014년에 VGG가 3 x 3 Convolution kernel과 기존보다 깊은 레이어를 쌓음으로써 높은 성능을 보이자, 레이어를 더 깊게 쌓는 많은 방식들이 제안되었습니다.
그 중 GoogLeNet은 22개의 레이어를 사용한 더 깊은 구조를 채용하여 ImageNet 2014에서 최고 성적을 거뒀는데요, 이 역시 Gradient Vanishing 문제로부터 자유롭지 않았습니다.
GoogLeNet에서는 이러한 문제를 해결하기 위해 Asymmetric Inception Module과 Auxiliary Classifier를 사용해 보았지만, 이 조차도 50개 또는 더 깊은 레이어를 쌓는 구조에서는 큰 효과를 거두기 어려웠습니다. 


<img width="320" alt="스크린샷 2022-08-10 오전 12 03 37" src="https://user-images.githubusercontent.com/52812351/183686047-1e861bfb-d923-4990-8118-4b95c5150fc9.png">


ResNet을 발표한 Microsoft Research에서도 레이어를 깊게 쌓을 수록 자연스럽게 low/mid/high level features들을 추출할 수 있고, 이렇게 추출한 feature들의 level 역시 레이어를 쌓을 수록 강화된다고 믿었습니다. 이러한 Deep Network의 장점을 살리기 위해서는 레이어가 깊어질수록 증가하는 training error를 어떻게 관리하느냐가 핵심이었는데요, Skip Connection을 활용한 Residual module로 성공을 거뒀습니다.

## ResNet의 특징

