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

#### ReLU 활성화 함수의 사용
  > 기존의 방식에서는 활성화 함수로 Sigmoid 함수와 tanh 함수를 주로 사용했습니다 (LeNet에선 tanh 함수만 사용). 하지만 경사 하강법을 기반으로 모델을 훈련시킬 때 이러한 saturating nonlinearities들은 입력값이 양극단으로 일정 수준 이상으로만 커지면 편미분을 진행할 때 dL/dw가 0에 가까이 수렴하게 되므로, w의 업데이트가 없어지는 saturation 현상, 더욱 더 나아가면 Gradient Vanishing 현상이 발생합니다. 때문에 ReLU와 같은 non-saturating nonlinearity보다 수렴이 어려울 수 밖에 없습니다. 
  실제로 동일한 조건에서 활성화 함수만 ReLU vs tanh로 설정하고 실험을 진행했을 때, Deep convolutional neural network는 ReLU 함수와 같이 작동했을 때 tanh보다 몇배 이상 빠른 속도를 냈다고 합니다. 
  
#### 다수의 GPU들을 활용한 model training
  > AlexNet test 당시에 활용된 GTX 580 GPU 한대에는 오직 3GB의 메모리 밖에 없어서 네트워크의 사이즈가 제한되었다고 합니다. 그래서 연구진들은 하나의 네트워크를 두 대의 GPU에 spread시켰고, 원활한 학습 진행을 위해 고도로 최적화된 2d Convolution 연산을 진행했다고 합니다.

#### 오버래핑 풀링
  > 전통적으로 풀링을 진행할 때는 stride와 kernel_size를 동일하게 하여 한 영역이 한번의 풀링만 적용되게 설정했습니다. AlexNet에서는 kernel_size를 stride보다 크게 설정하여
  (stride = 2, kernel_size = 3) 한 영역에 한번 이상의 풀링이 적용되게 설계하였습니다. 이러한 오버래핑 풀링의 사용으로 top-1과 top-5 error를 0.4%와 0.3%만큼 줄일 수 있었고, 오버래핑 풀링을 적용한 모델일수록 과적합의 리스크가 기존의 풀링 방식보다 줄어든다고 설명했습니다.
  
 #### Dropout의 차용
 > 다양한 모델들의 예측값을 종합해서 최종적인 예측값을 내는 방식은 test error를 줄이는데 매우 효과적인 방법입니다. 하지만 일정 사이즈 이상의 신경망은 훈련하는데만 며칠씩 걸리기 때문에 이와 같은 방법을 실제로 사용하기는 부적절 합니다. AlexNet과 비슷한 시기에 등장한 Dropout은 간단한 조정만으로도 매우 효과적으로 model combination 효과를 낼 수 있습니다. Dropout은 일정 확률로 각 뉴런의 값을 0으로 바꿔주는 방법입니다. 이렇게 "dropped out"된 뉴런은 순전파 과정에서 더 이상 관여를 하지 못하게 되고, 역전파 과정에서도 마찬가지 입니다. 즉, 입력이 들어올 때마다 신경망은 항상 조금씩 다른 형태를 취하게 됩니다. 이러한 방법은 뉴런간의 의존성을 줄일 수 있기 때문에 각각의 뉴런들이 보다 더 주도적으로 특징들을 학습할 수 있게하여 어떤 뉴런들과의 부분 조합을 형성하더라도 효과적으로 작동할 수 있게 도와줍니다.


## 모델 구조에 대해

* 입력: 3 x 224 x 224
* C1: k_size = 11, n_filters = 96, stride = 4
* Max_Pool: k_size = 3, stride = 2
* C2: k_size = 5, n_filters = 256, stride = 1
* C3, C4: k_size = 3, n_filters = 384, stride = 1
* C5: k_size = 3, n_filters = 256, stride = 1
* FC1, Fc2: out_size = 4096
* FC3: out_size = 1000



# Cat vs Dog 데이터셋으로 AlexNet 구현

## 이미지 데이터셋 전처리

<img width="575" alt="스크린샷 2022-08-08 오후 3 30 13" src="https://user-images.githubusercontent.com/52812351/183353749-de9439ec-3278-427d-bc7b-acf0e0d64e35.png">

AlexNet은 파라미터를 약 6000만개 사용하는 모델인만큼, 충분한 데이터가 없으면 과적합이 발생하는 등 테스트 데이터에 대한 성능이 좋지 않습니다. 데이터를 더 확보하기 위해 논문에서 적용한 horizontal reflection 이외에도 학습시에는 Random Resized Crop 및 Random Rotation을 확률적으로 적용하였고 validation 시에는 이미지 Resize와 Center Crop만 적용하였습니다. Augmentation 적용 후에는 이미지를 텐서 형태로 변환해주고 이미지넷 데이터의 평균과 표준편차를 사용해 정규화를 시켰습니다.


## DataLoader

<img width="677" alt="스크린샷 2022-08-08 오후 3 34 41" src="https://user-images.githubusercontent.com/52812351/183354336-bfd3cf21-7c5f-4b7b-ab51-13b87100ac00.png">
<img width="234" alt="스크린샷 2022-08-08 오후 3 34 53" src="https://user-images.githubusercontent.com/52812351/183354367-d9c2d136-2d31-4043-9ef7-354eb234b645.png">

논문에서는 128로 batch_size를 설정했지만 kaggle에서 지원하는 GPU의 성능상 batch_size를 32로 설정했습니다.
iterator를 통해 input의 사이즈를 추출해본 결과, [32, 3, 256, 256]으로 의도대로 나온것을 확인할 수 있었습니다.

## 모델의 네트워크 클래스 정의

<img width="561" alt="스크린샷 2022-08-08 오후 3 38 59" src="https://user-images.githubusercontent.com/52812351/183355020-7f5428af-0e0d-4ffd-8f5c-fb8ca22724bd.png">

위에서 언급한 것처럼, AlexNet은 첫번째 Convolutional Layer부터 두개의 GPU로 Filter들을 분담해가면서 연산을 진행합니다. 하지만 하나의 GPU만을 통해 구현을 하였기 때문에 임의로 합성곱 신경망의 filter size를 수정했으나, kernel size와 stride와 같은 조건은 일치시켰습니다. 

논문에서 언급한것과 같이 첫번째와 두번째 FC 레이어에 대해 Dropout을 적용해주었고, Feature Extractor와 Classifier 를 이어주기 위해 AdaptiveAvgPooling을 사용하여 한 필터마다 6x6 사이즈로 서브샘플링을 진행했습니다.



