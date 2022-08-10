Pix2Pix with facades dataset
====================================================

## Pix2Pix는 conditional GAN을 활용한 간단한 image-to-image translation으로써, 다양한 task에 공통적으로 적용할 수있는 generic approach로 사용 가능합니다.


## Pix2Pix의 탄생 배경
일반적인 Vanilla GAN은 실제 이미지와 구분하지 못할 수준의 이미지를 생성하는 능력을 가췄지만, random noise를 입력으로 받아 이미지를 생성하는 메카니즘을 가졌기 때문에 우리가 원하는 image를 특정하여 생성 시킬 수는 없었습니다. 이러한 단점을 해결하기 위해 cGan(conditional GAN)이 탄생하였는데 GAN이 입력 데이터에 대한 generative model을 학습한다면 cGAN은 입력 데이터에 대한 조건부 generative model을 학습합니다. 

Pix2Pix에선 conditional GAN을 활용하여 style transfer를 수행합니다. 기존에도 다양한 style transfer 방식들이 존재했지만, 각각의 방식이 특정 representation을 다른 representation으로 바꾸는데 최적화 되어 있어 범용성이 좋지 않았습니다. Pix2Pix에선 이러한 style transfer을 다양한 task에 공통적으로 적용할 수 있는것을 목표로 개발되었고, 이러한 특성을 반영하여 까다로운 손실 함수 및 하이퍼 파라미터 조정이 필요하지 않습니다.


## Conditional GAN에 대하여
cGan이랑 기존 Vanilla GAN에서 데이터의 모드를 제어할 수 있도록 조건 정보를 함께 입력하는 모델입니다.

아래와 같은 목적 함수를 가지고 있어서 Discriminator가 단순히 입력이 원본 데이터에 가까운지에 대해서만 판별하는게 아니라 얼마나 주어진 조건과도 비슷한지 판별해야 하고, Generator 역시 이러한 조건을 반영하여 이미지를 생성해내야 합니다.

<img width="1066" alt="스크린샷 2022-08-11 오전 12 34 40" src="https://user-images.githubusercontent.com/52812351/183945904-2fb3c2c1-1515-4ecf-a9c3-75484398bf88.png">

이러한 조건을 바탕으로, 사실적인이지만 랜덤한 숫자를 생성했던 vanilla GAN과 달리 아래와 같이 원하는 데이터의 조건을 함께 주어 원하는 category의 output을 얻을 수 있습니다.

<img width="1134" alt="스크린샷 2022-08-11 오전 12 36 49" src="https://user-images.githubusercontent.com/52812351/183946409-a06441b4-74ab-4923-9708-b5da54aee51c.png">


## Pix2Pix의 특징
우선 Image to Image translation이란, 특정한 도메인의 이미지를 다른 도메인으로 바꾼다는 것입니다. Pix2Pix는 학습 과정에서 이미지 x 자체를 조건으로 입력받는 cGAN의 한 유형으로, pixel들을 입력으로 받아 pixel들을 예측한다는 의미를 가집니다.

#### No more random noise, but image as an input
  > Vanilla GAN과 달리, Pix2Pix의 Generator는 입력으로 random noise를 받는 대신, 실제 이미지 x 자체를 조건으로 입력받아 다른 도메인의 이미지로 출력합니다. 
  > 이를 활용하여 입력으로 들어온 이미지의 key feature들은 유지하면서 전반적인 style만 mapping 하고픈 style로 변환할 수 있는데요, 반대로 생각해보면 random noise z를 사용하지 않기 때문에 거의 deterministic한 결과 위주로 생성할 수 밖에 없습니다. 물론 이러한 문제는 Dropout을 적용하여 sampling을 할 때마다 결과값이 바뀌도록 유도할 수 있습니다.
 
   <img width="504" alt="스크린샷 2022-08-11 오전 12 47 04" src="https://user-images.githubusercontent.com/52812351/183950503-ab2a9413-524c-4a85-a7b0-2d91421f663d.png">

#### U-Net 기반 아키텍쳐


> U-Net 아키텍쳐는 low-level information들을 skip connection을 이용해 네트워크를 가로질러 출력단에 가까운 Layer로 이동할 수 있게 해줍니다. 이미지를 조건으로 
입력받는 Pix2Pix의 특성상 input과 output 사이에서 low-level information을 공유하는것은 매우 중요한데, 아래의 실험 결과와 함께 살펴보겠습니다. 동일한 이미지에 대해 인코더-디코더와 U-Net 구조를 통해 이미지를 생성했는데, 여기서 인코더-디코더는 U-Net에서 skip connection만 제거한 구조입니다. 보시다시피 skip connection을 제거한 상태로 설계한 인코더-디코더는 사실적인 이미지를 생성하는데 실패하였습니다. 

 <img width="505" alt="스크린샷 2022-08-11 오전 12 52 44" src="https://user-images.githubusercontent.com/52812351/183954132-9618db25-2d31-4ec0-a7ec-ac58b809e7eb.png">

 <img width="781" alt="스크린샷 2022-08-11 오전 1 19 38" src="https://user-images.githubusercontent.com/52812351/183961749-1a31ee91-d879-4567-abf4-4eefafe3fbb4.png">

#### 손실 함수
> GAN은 자체적으로 이미지의 진위 여부를 판단하기 때문에, 다른 생성 모델에 비하여 blurry한 output이 나오는 문제가 적은 편입니다. 하지만 GAN의 성능을 더욱 더 향상시키기 위해, 기본적인 GAN의 목적 함수 뿐만 아니라 L1 loss 함수를 함께 사용하여 현실적인 이미지를 만들 뿐만 아니라 실제 정답과 유사하도록 학습을 시킵니다. L1 loss를 이용하는 이유는 L2 loss를 사용할 때보다 blurry 현상이 덜 발생하였고, Eucledian distance (MSE Loss)를 활용하지 않는 이유는 MSE는 output 결과들에 대한 평균을 만들며 miminimze되는데, 이러한 과정이 blurring을 야기하기 때문입니다.

<img width="431" alt="스크린샷 2022-08-11 오전 1 26 21" src="https://user-images.githubusercontent.com/52812351/183963075-fe883b0a-c071-440e-b20d-5145c253f72d.png">



#### PatchGAN 활용
> 일반적인 GAN 모델들이 Discriminator가 입력으로 들어온 이미지의 전체를 보고 그 이미지의 진위 여부를 판단하는 것과 달리, Pix2Pix에서는 Discriminator가 convolutional PatchGAN classifier의 형태를 띄게 합니다. PatchGAN이란 이미지 전체에 대하여 판별하지 않고, 이미지를 여러개의 패치로 나눈 다음에, 패치 단위로 진위 여부를 판단합니다. 논문에서는 보다 High-frequency한 이미지를 모델링하기 위해서 attention을 local image patches로 restrict하는 것이 합당하다고 주장합니다. 이러한 PatchGAN의 활용은 보다 적은 parameter수로 이어지고 더 빨리 동작하며, 임의의 큰 사이즈의 이미지에서도 잘 동작할 수 있게 해준다고 논문에서는 주장하였습니다.


## Facades 데이터셋을 활용한 Pix2Pix 구현

### 이미지 전처리 및 커스텀 데이터셋 설계

<img width="539" alt="스크린샷 2022-08-11 오전 1 46 55" src="https://user-images.githubusercontent.com/52812351/183967293-ca6a2145-0a2a-4a95-b7db-511c4c40a52b.png">

논문에서도 사용된 Facades 데이터셋을 활용하여 labels -> photo image translation을 구현해 보았습니다.

<img width="389" alt="스크린샷 2022-08-11 오전 1 48 46" src="https://user-images.githubusercontent.com/52812351/183968045-acbe6b6f-9b6a-4f52-8492-e14a457dc966.png">

우선 학습용으로 주어진 이미지 하나를 불러와서 이미지의 형태를 살펴보았습니다. Photo에 해당하는 이미지가 왼쪽에 있고, Paint에 해당하는 이미지가 그 오른쪽에 붙어있는 형식의 이미지었고 각 이미지의 크기는 256 x 256이었습니다. 

<img width="625" alt="스크린샷 2022-08-11 오전 1 50 29" src="https://user-images.githubusercontent.com/52812351/183968787-4fe36cbc-38f7-4e0a-920b-cd1204da9df8.png">

때문에 커스텀 데이터셋을 설계할 때 위와 같이 원본 이미지와 조건 이미지를 분리해 주었고, 각각 Real과 Condition으로 반환시켜 주었습니다.


<img width="616" alt="스크린샷 2022-08-11 오전 1 52 36" src="https://user-images.githubusercontent.com/52812351/183969604-7a0e207d-c10a-4819-91bc-4bbf7c271b29.png">

이미지들은 256 x 256 사이즈로 Resize 시켜주었고, 이 때 BICUBIC 보간법을 적용해주었습니다. 텐서로 변환해 주었고, 이미지 픽셀의 범위를 -1 ~ -1로 정규화 시켜주었습니다. train과 val dataloader 모두 배치 사이즈는 10으로 설정해 주었습니다.

### Generator 설계

<img width="563" alt="스크린샷 2022-08-11 오전 1 57 51" src="https://user-images.githubusercontent.com/52812351/183971785-dbba39f8-f958-476e-8082-692a7c9ec56c.png">

논문에서 언급한대로 U-Net 아키텍쳐를 기반으로, Downsampling과 Upsampling을 순서대로 시행하였습니다. Downsampling을 시행할 때마다 이미지의 Height와 Width를 2배씩 감소 시켰고, 반대로 Upsampling을 시행할 때는 2배씩 증가시켰는데 이 과정에서 Skip Connection을 활용하여 동일한 차원을 가지는 DownSampling layer의 결과값을 concatenation 시켜주어서 low level feature를 더해줬습니다. Upsampling을 진행할 때 일반적인 보간법이 아닌 Transposed Convolution을 활용하여 이 또한 학습 가능하게 설계하였습니다. 마지막에는 출력의 크기를 [3, 256, 256]로 맞춰 이미지를 반환합니다.


### Discriminator 설계

<img width="804" alt="스크린샷 2022-08-11 오전 2 04 05" src="https://user-images.githubusercontent.com/52812351/183973402-5235a935-ee98-4705-a3b1-9b3f12cba600.png">

 Discriminator는 Real image와 Condition image 이 둘을 입력으로 받기에 그 둘을 torch.cat을 활용하여 channel 단위로 합쳐줬습니다. 이를 바탕으로 매번 이미지의 Height와 Width를 두배씩 감소시켰고, 마지막에는 논문에서 언급한것과 같이 PatchGan을 적용시키기 위해 (1, 16, 16) 사이즈로 합성곱 연산을 통해 만들어준 다음 반환시킵니다.
 
 
 ### 모델 초기화 및 손실 함수, 학습률, 최적화 함수 설계
 
 <img width="709" alt="스크린샷 2022-08-11 오전 2 10 06" src="https://user-images.githubusercontent.com/52812351/183974488-9ff31743-eba0-483e-b313-ccc798e464ac.png">
 
 논문에서 언급한대로, 가우시안 분포를 따르는 (0.0, 0.02)로 가중치 초기화를 진행해주었습니다. 생성자와 판별자를 선언하고 gpu를 할당했고, 일반적인 GAN 부분에 대해선 MSELoss를 사용했고, 보다 이미지가 ground truth와 가까워지게 L1Loss를 추가로 사용했습니다. 학습률은 0.0002로 설정해두었고, 생성자와 판별자 모두 ADAM 옵티마이저를 사용했습니다.
 
 
 ### 모델 학습
 
 <img width="653" alt="스크린샷 2022-08-11 오전 2 58 33" src="https://user-images.githubusercontent.com/52812351/183983641-3bdb1191-3700-4441-bacd-eeeaec003e7c.png">

우선 patchGAN을 활용하여 Discriminator가 patch 단위로 이미지의 진위 여부를 판단할 수 있게 real과 fake를 알맞은 크기로 선언해주었습니다.

#### Generator 학습
> Generator에 condition image를 입력으로 주어 fake_Photo를 생성했고, 이를 사용하여 Discriminator가 fake Photo와 real Paint를 동시에 입력으로 받았을 때 해당 Photo의 진위 여부를 판단할 수 있게 설정하였고, 실제 정답과 유사하도록 L1 Loss를 사용하여 fake Photo와 real Photo간의 차이를 loss에 추가해 주었습니다.

#### Discriminator 학습
> Discriminator는 real_Photo와 real_Paint가 동시에 주어졌을 때 해당 Photo가 주어진 조건인 real_Paint의 key feature를 잘 가지고 있는 Photo인지 판별하는 학습 능력을 기르고, 반대로 real_Paint를 통해 생성자가 생성한 fake_Photo를 잘 구분할 수 있는지 학습합니다.


### 학습 결과
세로순으로 조건 이미지 -> 생성된 이미지 -> 원본 이미지 입니다.
전반적으로 학습이 진행될수록 원본 이미지와 유사하게 이미지가 생성되었음을 확인할 수 있습니다.

<img width="799" alt="스크린샷 2022-08-11 오전 3 04 37" src="https://user-images.githubusercontent.com/52812351/183984794-8d4e931a-0d2e-4fe9-b919-7e9d6fcc8170.png">
<img width="803" alt="스크린샷 2022-08-11 오전 3 04 49" src="https://user-images.githubusercontent.com/52812351/183984846-3b70347c-1dcd-4fa8-ba24-4971aeadc6f5.png">
<img width="804" alt="스크린샷 2022-08-11 오전 3 04 59" src="https://user-images.githubusercontent.com/52812351/183984879-08c81973-5389-40f9-8994-a94d6183b2e4.png">







