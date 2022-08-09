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
Convolutional Layer는 대부분 3 x 3 filter size를 가지고 있으며, 아래의 원칙들을 따랐습니다.

* 같은 output feature map size를 가진 경우, 같은 필터 갯수를 같도록
* 만약 feature map size가 반으로 줄었다면, 필터 갯수를 두배로 늘려서 레이어별 시간 복잡도를 유지하도록.
* 정해진 횟수마다 skit connection을 추가하도록,
* 필터 갯수가 늘어서 feature map size가 줄었다면, skip connection일때도 이것에 맞춰주기 위해 1 x 1 convolution으로 차원을 맞춰준다.
* Convolution Layer 이후에 Batch Normalization 레이어 추가.

사용된 최적화 알고리즘은 다음과 같습니다.

* Mini-Batch Gradient Descent with batch size of 256
* Momentum 0.9
* Wedight decay of 0.0001
* learning rate starting from 0.1 and deviced by 10 when the error plateaus.


<img width="896" alt="스크린샷 2022-08-10 오전 1 49 30" src="https://user-images.githubusercontent.com/52812351/183710518-830cc9b6-d1a6-477c-9546-5e0f69bef721.png">

전반적으로 위와 같은 구조를 지녔습니다.


## CIFAR-10 데이터셋으로 ResNet-18, 34 구현

<img width="439" alt="스크린샷 2022-08-10 오전 1 51 27" src="https://user-images.githubusercontent.com/52812351/183710888-9d002bdc-883a-459c-b507-303f67661790.png">

ResNet-18과 ResNet-34가 convn_x layer마다 사용된 convolutional layer의 갯수만 다를 뿐 전반적으로 공통된 구조를 공유하고 있다는걸 확인한 후, VGG를 구현할때와 마찬가지로 비슷하게 모듈화하여 구현하되, convx_n 레이어의 convolutional layer 갯수만 정해진 모델에 따라 바뀔 수 있게 설계해 보았습니다.

이번에는 코드를 기능에 따라 나누어 모듈화 해보았습니다.

## main.py

* arparse를 사용하여 다양한 하이퍼 파라미터들을 튜닝할 수 있게 설정했습니다.
* datasets.py 파일에 접근하여 dataloader를 받아온 다음에, 설정한 batch_size만큼 데이터를 불러오는 trainloader, testlader를 선언했습니다.
* 이를 training.py 파일의 train_model 함수에 넘겨주었습니다.
* 현재 train 모드인지 eval 모드인지에 따라 해당하는 작업을 진행하도록 설정했습니다.

## datasets.py

* 학습시 사용할 전처리의 경우 RandomCrop, RandomHorizontalFlip을 확률적으로 적용한 후 텐서로 변환해주고 CIFAR-10 데이터에 대한 평균과 표준편차를 적용하여 정규화를 진행했습니다.
* 테스트때 사용할 전처리의 경우, 텐서로 변환만 해주고 위와 같은 전처리만 적용 했습니다.
* 이후엔 목적에 맞는 전처리를 적용하면서 데이터를 불러오고,
* 각각 DataLoader를 통해 반환해주는 작업을 거치도록 설계했습니다.

## models.py

> ### ResidualBlock(nn.Module):
> 우선 Residual Block을 정의해줬습니다. 전반적으로 Residual Block이 두개의 Convolutional Layer 이후에 입력값을 skip connection으로 더해주는 방식이었던 것을 고려하여 conv_block과 downsample로 나누어 구성해줬습니다. 
>  > Conv_block에 대해선 두개의 Convolution Layer를 입력값으로 들어오는 in_channels, out_channels에 맞게 설정해 주었고, 배치 정규화와 ReLU 함수를 추가했습니다.

>  > Stride가 1이 아니거나 in_channels != out_channels라면, 변화된 크기에 대응해주기 위해 1 x 1 Convolution을 수행하여 다운 샘플링을 진행했습니다.

> #### ResNet(nn.Module):
> 원 논문에서는 Residual Block 이전의 conv1에서 7 x 7, 64, stride 2의 Convolution 연산과, 3 x 3 max pool, stride 2를 진행해 주었는데요, CIFAR-10 데이터셋 특성상 위와 같은 연산을 동일하게 적용할 경우 이미지 사이즈가 너무 작아질것을 우려하여 kernel size를 3으로 조정해주었고, Max Pooling 과정을 생략했습니다.
> conv2_x, conv3_x, conv4_x, conv5_x에 대해서는 make_layer라는 내부 함수를 작성해서 상황에 맞는 Residual Block을 알맞은 갯수로 생성할 수 있게 하였습니다.

>> #### modeltype(model):
>> 터미널 단에서 model 이름을 입력으로 받아, 해당 ResNet 구조에 맞는 Residual Block 덩어리의 갯수를 리스트 형태로 포함하여 ResNet 함수에 ResidualBlock과 함께 전달하여 네트워크를 생성하게 설계했습니다.




