VGG with Cat vs Dog classificatio problem
====================================================

## VGGNet은 CNN 레이어를 기존에 제시되었던 방식보다 더욱 깊게 쌓으며 모델의 깊이와 성능의 상관관계를 직접적으로 보여줬습니다.


## VGGNet의 탄생 배경
AlexNet의 성공 이후, 컴퓨터 비전 영역에서는 AlexNet을 더욱 더 발전시키고자 많은 노력들이 이루어졌습니다. 
예시로, ILSVRC-2013에서 우승한 모델은 첫번째 Convolutionaly Layer에서 보다 작은 kernel_size와 stride를 사용했습니다.
VGG는 무엇보다 ConvNet 아키텍쳐의 '깊이'에 초점을 맞췄습니다. 기본적인 설계 구조는 AlexNet의 그것을 따라갔지만, AlexNet에서 학습 가능한 Layer가 8개에 불과했던 반면 VGG에서는 학습 가능한 Layer의 개수가 11개부터 19개까지 기존의 네트워크보다 훨씬 깊은 네트워크 구조를 가지고 있습니다.

## VGG의 특징

#### 3x3 receptive field
  > VGG 이전에 Convolutional Network를 활용하여 이미지 분류에서 좋은 성과를 보였던 모델들은 비교적 큰 사이즈인 11 x 11, 혹은 7 x 7 receptive field를 사용했습니다. 그러나 VGG에서는
  3 x 3 사이즈의 작은 receptive field만을 사용합니다. 논문에 따르면 이것이 좌/우, 상/하, 그리고 중앙의 개념을 캡쳐할 수 있는 가장 작은 사이즈이기에 3 x 3 사이즈를 채택했다고 합니다.
  
  <img width="302" alt="스크린샷 2022-08-09 오후 4 59 30" src="https://user-images.githubusercontent.com/52812351/183596595-ad68e414-1426-411c-90fb-8a8f0831e86f.png"> <img width="339" alt="스크린샷 2022-08-09 오후 4 59 41" src="https://user-images.githubusercontent.com/52812351/183596633-4d1bed4d-8c5f-457e-bd2d-ad9fda8ddd58.png">

  > 위와 같이 10 x 10 사이즈가 있다면, Stride = 1로 고정되어 있을 때 3차례의 3 x 3 Convolution 연산을 반복한 Feature Map은 7 x 7 Convolution 연산을 한번 적용했을 때와 같은 효과를 볼 수 있습니다. 그렇다면 3 x 3 Convolution Layer를 3번 반복함으로써 7 x 7 Convolution Layer를 한번 적용한는 것에 비해 어떤 장점이 있을까요?
  >   > #### 결정 함수의 비선형성 증가
