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

> #### Residual Learning
> ResNet에서는 Residual Block를 이용해 네트워크의 최적화 난이도를 낮춥니다. 일반적으로는 네트워크가 입력 데이터인 x를 레이어에 통과시키면서 입력값을 최적의 결과값으로 매핑할 수 있는 H(x)로 fit하길 바랍니다만, 실제로 내재한 매핑인 H(x)를 앞선 layer에서 학습한 정보를 포함하여 곧바로 학습하는 것은 어려우므로 대신 F(x) = H(x) - x를 학습시킵니다. 극단적인 예로, 만약에 identity mapping인 x가 우리가 원하는 optimial이라면, F(x)를 0으로 수렴시키는 방법이 여러개의 비선형 레이어를 통과하면서 identity mapping을 수행하는 것보다 훨씬 쉽습니다.  

<img width="214" alt="스크린샷 2022-08-10 오전 12 21 12" src="https://user-images.githubusercontent.com/52812351/183689753-e39e4007-2930-407c-a760-3b413c9b20cf.png">

<img width="380" alt="스크린샷 2022-08-10 오전 12 35 37" src="https://user-images.githubusercontent.com/52812351/183695207-8127dd54-4a0e-4f71-990b-f84cc75de4b7.png">

> 앞선 Layer에서 학습된 정보인 x를 그대로 identity mapping 해줌으로써, 해당 레이어는 자신 뿐만아니라 이전의 모든 네트워크를 고려해서 학습하기보단, F(x)에 해당하는 범위만 학습할 수 있기 때문에 부담을 덜 수 있고, 입력값 x를 identity mapping으로 감싸서 정보가 손실되지 않게 출력값으로 전달할 수 있습니다. 또한, 이러한 Skip Connection을 적용하여 결과적으로 해당 Residual Block의 식이 F(x) + x가 되고 F(x)가 0이 되는 방향으로 수렴한다면, 미분값은 F'(x) + 1이기 때문에 네트워크가 깊더라도 이전보다 안정적으로 학습이 가능해집니다. 

> 또한, F(x)가 0으로 학습하는 과정에서 단순히 0이 되는 것이 아니라 0으로 학습하면서 기존 CNN과 유사하게 입력값의 비선형적인 특성을 학습할 수 있습니다. 


## 모델 구조에 대해

기본적으로 VGG의 구성을 따라가고 있습니다.
Convolutional Layer는 대부분 3 x 3 filter size를 가지고 있으며, 아래의 두 원칙을 따랐습니다.

* 같은 output feature map size를 가진 경우, 같은 필터 갯수를 같도록
* 만약 feature map size가 반으로 줄었다면, 필터 갯수를 두배로 늘려서 레이어별 시간 복잡도를 유지하도록.
* 정해진 횟수마다 skit connection을 추가하도록,
* 필터 갯수가 늘어서 feature map size가 줄었다면, skip connection일때도 이것에 맞춰주기 위해 1 x 1 convolution으로 차원을 맞춰준다.


