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

#### No more random noise, but an image itself
  > Vanilla GAN과 달리, Pix2Pix의 Generator는 입력으로 random noise를 받는 대신, 실제 이미지 x 자체를 조건으로 입력받아 다른 도메인의 이미지로 출력합니다. 
  > 이를 활용하여 입력으로 들어온 이미지의 key feature들은 유지하면서 전반적인 style만 mapping 하고픈 style로 변환할 수 있는데요, 반대로 생각해보면 random noise z를 사용하지 않기 때문에 거의 deterministic한 결과 위주로 생성할 수 밖에 없습니다. 물론 이러한 문제는 Dropout을 적용하여 sampling을 할 때마다 결과값이 바뀌도록 유도할 수 있습니다.
 
  > <img width="504" alt="스크린샷 2022-08-11 오전 12 47 04" src="https://user-images.githubusercontent.com/52812351/183950503-ab2a9413-524c-4a85-a7b0-2d91421f663d.png">

