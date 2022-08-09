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
  >   > 3 x 3 Convolution Layer를 3번 적용하게 되면, 3번의 ReLU 함수를 거치게 되지만, 7 x 7 Convolution Layer를 한번 적용하게 되면 단 한번의 ReLU 함수밖에 거치지 못하게 됩니다. 이러한 차이는 비선형성을 더욱 증가시키고, 결과적으로 모델의 특징 식별성 증가로 이어집니다.

>    > #### 학습 파라미터 수의 감소
>    > Fully Connected Layer와는 달리 Convolutional Network의 경우, 학습 대상인 가중치는 필터의 크기에 따라 결정됩니다. 하나의 Layer에 하나의 필터만 있다고 가정할 경우, 7 x 7 Convolution Layer 1개에 대한 학습 파라미터 갯수는 7 x 7 = 49이지만, 3번의 3 x 3 Convolution Layer의 학습 파라미터 갯수는 3 x 3 x 3 = 27입니다. 즉, 3 x 3 Convolution Layer를 겹쳐서 적용할 경우 7 x 7 Convolution Layer를 한번 적용한것에 비해 약 81% 적은 학습 파라미터 수를 가지며 더욱 빠른 학습 속도와 과적합 예방 효과를 불러올 수 있습니다.


#### 가중치 초기화
  > 논문에서는 잘못된 가중치 초기화는 네트워크가 깊어질 수록 불안정한 gradient에 의해 발생되는 Gradient Vanishing/Exploding을 야기할 수 있기 때문에 올바른 가중치 초기화가 중요하다고 언급했습니다. 이러한 문제를 해결하기 위해 논문에서는 상대적으로 얕은 깊이를 가진 ConvNet configuration Table에서 configuration A의 weight들을  평균이 0, 표준편차가 0.01인 정규분포를 따르는 분포를 바탕으로 random initialise 시키고 학습을 진행하였습니다. 그리고 더욱 깊은 네트워크를 학습할 때 해당 네트워크의 첫 4개의 Convolutionlay Layer와 마지막 3개의 Fully Connected Layer의 weight를 미리 학습해둔 configuration A의 모델의 해당 레이어에 대한 가중치로 초기화 시켜줬습니다. 하지만 논문 제출 후, Glorot 초기화를 사용하면 이러한 사전 훈련 과정 없이 random initialisation을 사용할 수 있다는 것을 깨달았다고 합니다.



## 모델 구조에 대해

<img width="395" alt="스크린샷 2022-08-09 오후 6 09 29" src="https://user-images.githubusercontent.com/52812351/183610916-1b6368c3-ab30-4b8a-abd9-dadd21fe2fa1.png">

VGG 네트워크는 깊이에 따라 위와 같이 세분화 되어 있습니다. 공통적으로는,
* 1n Convolution Layers + 3 Fully-Connected Layer로 구성되어 있고,
* 3 x 3 convolution filter size
* Stride = 1, Padding = 1
* Maxpooling of 2 x 2 kernel size with stride of 2 (AlexNet의 Overpolling 방식을 차용하지 않았습니다
* ReLU

위와 같은 구조를 공유하고 있습니다.


<img width="395" alt="스크린샷 2022-08-09 오후 6 09 29" src="https://user-images.githubusercontent.com/52812351/183610916-1b6368c3-ab30-4b8a-abd9-dadd21fe2fa1.png">


