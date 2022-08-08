AlexNet with Cat vs Dog Classification problem
====================================================


## AlexNet은 2012년에 University of Toronto에서 발표된 논문입니다. 더욱 더 발전된 하드웨어와 GPU 연산 device를 바탕으로 더욱 깊은 레이어를 구성했고, 이를 바탕으로 ILSVRC 2012에서 우승했습니다.

## AlexNet의 탄생 배경
1998년에 LeNet이 CNN 기반의 네트워크로 MNIST Dataset에 대하여 99.05%의 정확도를 기록하며 가능성을 보여줬지만, 다양한 한계들이 존재했습니다. 
  > 대용량의 labeled dataset의 부재
  MNIST와 같은 Simple recognition tasks들은 비교적 적은 크기의 데이터셋으로도 충분히 높은 성능을 낼 수 있었지만, 현실에서 다루는 이미지들에는 다양성이 존재하고, 이를 학습하기 위해서는 기존에    존재하던 데이터셋의 크기보다 훨씬 큰 데이터셋이 필요했습니다. 

  > 부족한 연산 능력
  백만여가지의 이미지에서 천여가지의 객체를 탐지하고 분류하기 위해서는 large learning capacity를 가진 모델이 필요합니다. LeNet에서 다뤘던 합성곱 신경망은 이러한 능력을 가진 모델이고 지역적인    특징을 활용할 수 있는 멋진 모델이지만, 이를 고화질의 이미지들에 적극적으로 활용하기에는 하드웨어적인 비용이 매우 컸습니다. 때문에, AlexNet 논문 발표 직전에도 객체 인식 및 분류 문제에서는 머신러    닝 기법을 바탕으로 한 접근 방식이 널리 사용되었다고 합니다.

하지만 2012년이 되어, LabelMe, ImageNet 데이터셋을 비롯한 대용량을 labeled dataset들이 공개되고, 하드웨어의 성능이 올라감에 따라 기존의 LeNet 구조와 비교해서 더욱 깊고 세밀한 네트워크를 사용할 수 있게 되었습니다.


## AlexNet의 특징

### 총 8개의 학습 가능한 레이어로 구성된 아키텍쳐
  >AlexNet은 5개의 합성곱 레이어와 3개의 Fully Connected 레이어를 가지고 있어, 2개의 합성곱 레이어와 2개의 Fully Connected 레이어를 가지는 LeNet보다 깊은 아키텍쳐를 가지고 있습니다.
  
  
<img width="685" alt="스크린샷 2022-08-07 오후 5 23 29" src="https://user-images.githubusercontent.com/52812351/183282101-deafff15-7ecd-46ee-ae30-f3502fb93f74.png">

  > ReLU 활성화 함수의 사용
  기존의 방식에서는 활성화 함수로 Sigmoid 함수와 tanh 함수를 주로 사용했습니다 (LeNet에선 tanh 함수만 사용). 하지만 경사 하강법을 기반으로 모델을 훈련시킬 때 이러한 saturating nonlinearities들은 입력값이 양극단으로 일정 수준 이상으로만 커지면 편미분을 진행할 때 dL/dw가 0에 가까이 수렴하게 되므로, w의 업데이트가 없어지는 saturation 현상, 더욱 더 나아가면 Gradient Vanishing 현상이 발생합니다. 때문에 ReLU와 같은 non-saturating nonlinearity보다 수렴이 어려울 수 밖에 없습니다. 
  실제로 동일한 조건에서 활성화 함수만 ReLU vs tanh로 설정하고 실험을 진행했을 때, Deep convolutional neural network는 ReLU 함수와 같이 작동했을 때 tanh보다 몇배 이상 빠른 속도를 냈다고 합니다. 
  
  > 다수의 GPU들을 활용한 model training
  AlexNet test 당시에 활용된 GTX 580 GPU 한대에는 오직 3GB의 메모리 밖에 없어서 네트워크의 사이즈가 제한되었다고 합니다. 그래서 연구진들은 하나의 네트워크를 두 대의 GPU에 spread시켰고, 원활한 학습 진행을 위해 고도로 최적화된 2d Convolution 연산을 진행했다고 합니다.

  > 오버래핑 풀링
  > ㅇㅇ






