Generative Adversarial Nets with MNIST dataset
=================================================

## GAN은 생성자와 판별자 두 개의 네트워크를 활용한 생성 모델로, 매우 다양한 논문들을 파생시킨 대표적인 생성 모델입니다.


## GAN의 탄생 배경
사진, 음성 파일, 또는 자연어에서의 다양한 심볼들의 확률 분포를 나타낼 수 있는 rich하게 계층적인 모델을 만드는것은 딥러닝의 궁극적인 목표라고 할 수 있습니다. GAN 논문이 발표될 당시에만 하더라도, Discriminative models, 즉 classification 분야에서 개별 선형 유닛들을 gradient와 함께 이용하여 뛰어난 성과를 거뒀습니다만 Deep generative model은 상대적으로 영향력이 덜했습니다. 여기에는 다양한 확률론적 계산을 근사화하는 것이 어렵고 생성하는 측면에서 discriminative model의 개별 선형 유닛들을 활용함으로써 얻을 수 있는 이점을 얻지 못했기 때문입니다. 이러한 단점을 바탕으로, 2014년에 Montreal 대학교에서 새로운 생성 모델을 설계하는 방법일 제안했습니다.

## 생성모델이란?

<img width="768" alt="스크린샷 2022-08-10 오후 9 19 09" src="https://user-images.githubusercontent.com/52812351/183899313-80a57edf-d0a0-4c8a-be71-9e0f54928d20.png">

우선 컴퓨터가 어떻게 존재하지는 않지만 인간의 눈으로 봤을 때 충분히 그럴싸한 이미지를 만들어낼 수 있는지 알아보겠습니다. 확률 분포란 확률 변수가 특정한 값을 가질 확률을 나타내는 함수인데. 이미지 데이터는 다차원 특징 공간의 한 점으로 표현되는데, 이러한 다양한 특징들이 각각의 확률 변수가 되어 다차원 확률 분포를 이룹니다. 이러한 이미지의 분포를 근사하는 모델을 학습하게 된다면 충분히 사실적인 이미지를 표현할 수 있게됩니다. 예를 들어서 사람의 얼굴에는 통계적인 평균치가 존재할 수 있는데, 모델은 이를 수치적으로 표현할 수 있게 됩니다. 생성 모델은 실존하진 않지만 있을법한 이미지를 생성할 수 있는데, 위와 같은 사람의 얼굴 이미지를 근사한 다변수 확률 분포에서 확률이 높은 부분의 데이터를 추출할수록 실제로 존재할법한 사람의 얼굴 이미지를 생성할  수 있게 됩니다.

다시 한번 정리하자면, 원본 이미지 데이터의 분포를 근사하는 모델 G를 만드는 것이 생성 모델의 목표입니다. 모델 G가 잘 작동한다는 의미는 원래 이미지들의 분포를 잘 모델링할 수 있다는 것을 의미하는데, 학습이 잘 되었다면 통계적으로 평균적인 특징을 가지는 데이터를 쉽게 생성할 수 있습니다. 

## GAN의 특징

#### Adversarial Network
> GAN을 처음 제안한 Ian Goodfellow는 GAN을 생성자가 판별자를 속이고, 반면에 판별자는 판별하는 대상이 원본인지 가품인지 판단하는 minimax game에 비유할 수 있다고 설명했습니다. 더욱 더 직관적으로는 경찰(판별자)과 위조지폐범(생성자)사이의 게임에 비유하였는데요, 위조지폐범은 최대한 진짜 같은 화폐를 생성하여 경찰을 속이기 위해 노력하고, 경찰은 진짜 화폐와 가짜 화폐를 완벽히 판별하여 위조지폐범을 검거하는 것을 목표로 합니다. 이러한 경쟁적인 학습이 지속되다 보면 어느 순간 위조지폐범은 진짜와 다를 바 없는 위조 지폐를 만들 수 있게 되고, 경찰이 위조지폐를 구별할 수 있는 확률도 가장 헷갈리는 50%로 수렴하게 되어 경찰은 위조지폐와 실제 화폐를 구분할 수 없는 상태에 이르게 됩니다. 

<img width="694" alt="스크린샷 2022-08-10 오후 9 23 40" src="https://user-images.githubusercontent.com/52812351/183900192-80fde0a7-e996-46bf-a84c-41302541e669.png">


> Adversarial Network에서는 분류 모델과 생성 모델을 학습시키는 과정을 서로 주고받으면서 반복합니다.
##### Discriminitive Model
> > 분류 모델의 학습은 크게 두 가지 단계로 이루어져 있습니다.
> > * 진짜 데이터를 입력해서 네트워크가 해당 데이터를 진짜로 분류하도록 학습하는 과정.
> > * 반대로, 생성 모델에서 생성한 가짜 데이터를 입력해서 해당 데이터를 가짜로 분류하도록 학습하는 과정.
> > 이 과정을 통해 분류 모델은 진짜 데이터를 진짜로, 가짜 데이터를 가짜로 분류할 수 있게 됩니다.

#### Generative Model
> > 그렇다면 생성 모델은 위와 같이 학습된 분류 모델을 속이는 방향으로 학습되어야 합니다.
> > * 생성 모델에서 만들어낸 가짜 데이터를 판별 모델에 입력하고, 가짜 데이터를 진짜라고 분류할 만큼 진짜 데이터와 유사한 데이터를 만들어 내도록 생성 모델을 학습시킵니다.

#### Random Noise
> 이러한 프레임워크는 다양한 방법들을 통해 구현될 수 있는데, GAN 논문에서는 Generative Model에게 random noise를 multilayer perceptron을 거쳐서 넘겨주는 방식으로 sample을 생성하는 방식을 채용했습니다. 아래의 목적 함수에서 z가 noise인데, P_z(z)라는 z의 확률 분포에서 임의로 하나의 sample을 추출하여 이를 Generative Model에 전달하여 가짜 이미지를 생성하고 Discriminative Model로 하여금 해당 이미지의 진위 여부를 판단하게 합니다.

>   <img width="1007" alt="스크린샷 2022-08-10 오후 9 32 02" src="https://user-images.githubusercontent.com/52812351/183901925-59fe179a-b127-452c-9408-c4b80f43552c.png">

최종적으로 Generative Model을 통해 생성된 데이터의 분포가 원본 이미지들의 분포를 따르게 되는것이 목표입니다.

<img width="142" alt="스크린샷 2022-08-10 오후 9 42 47" src="https://user-images.githubusercontent.com/52812351/183904050-cc763851-96af-4cf2-bfcc-ab5f8322b9de.png">

#### 실험 결과
생성 모델이 우연히 정말로 그럴싸한 데이터를 만들어 내는 것인지, 아니면 정말로 데이터를 완벽하게 이해하고 있어서 활용할만한 가치가 있는 모델인지 알아보는것은 매우 중요한 evaluation 과정입니다. 
생성자의 메커니즘을 조금 더 들여다보면, 생성자 G의 입력으로 latent vector인 z가 들어가게 됩니다. G의 출력이 사람의 얼굴이라고 했을 때, 왼쪽을 바라보는 얼굴을 만들어 내는 z1의 평균 벡터와 오른쪽을 보고 있는 얼굴을 만들어 내는 z2의 평균을 계산하고 이 두 벡터 사이의 축을 중간에서 interpolation하여 생성자로 입력하면 천천히, 그리고 자연스럽게 회전하는 얼굴이 나오는 것을 확인할 수 있습니다.

<img width="762" alt="스크린샷 2022-08-10 오후 10 10 16" src="https://user-images.githubusercontent.com/52812351/183909692-d82b85cc-be98-49e0-8311-be56f7b15a2a.png">

또한, 논문에서는 이러한 결과가 의도적으로 가장 좋은 성능을 보이는 부분만 가져온것이 아니고, training set을 단순히 암기하여 이와 같은 결과를 내는것이 아님을 강조하였습니다.



## MNIST를 이용한 GAN 실습

### Discriminator 설계

<img width="372" alt="스크린샷 2022-08-10 오후 10 24 11" src="https://user-images.githubusercontent.com/52812351/183912486-921bbd91-bb20-4d5f-bbb0-d5bbb1165ad8.png">

MNIST Dataset의 이미지 한 장의 크기가 [1, 28, 28]인것을 고려하여 그것에 맞게 Linear Layer로 입력을 받았고, 512 -> 256 -> 그리고 1로 점진적으로 downsampling 시켜줬습니다. Layer 중간마다 LeakyReLU 활성화 함수를 사용해 주었고, 마지막에는 입력으로 들어온 해당 이미지가 실제 데이터셋의 이미지인지 0~1의 확률로 표현할 수 있게 Sigmoid 활성화 함수를 붙여줬습니다.


### Generator 설계

<img width="133" alt="스크린샷 2022-08-10 오후 10 28 17" src="https://user-images.githubusercontent.com/52812351/183913367-a9dbfd6d-d08a-41d8-aa93-73efabe8694a.png">

<img width="493" alt="스크린샷 2022-08-10 오후 10 28 05" src="https://user-images.githubusercontent.com/52812351/183913329-3ae3611a-6d02-4f06-88ce-2ad09140f052.png">

> 우선 random noise인 z의 latent dimension으로 100을 설정해주었습니다. layer 설계를 block 단위로 하기 위해 입력 차원과 출력 차원을 받아 해당하는 Linear Layer를 만들고 필요해 따라 정규화를 진행하며 LeakyReLU 활성화 함수까지 추가해주는 Block 함수를 만들었습니다. 이를 바탕으로 처음엔 100의 latent_dim 값 -> 128 -> 256 -> 512 -> 1024의 업샘플링을 거치고, 마지막엔 MNIST 데이터셋 이미지 한개의 사이즈인 1 * 28 * 28로 변환해주고 Tanh 활성화 함수를 통과하게 하여 -1 ~ 1의 값을 가지게 하였습니다. 
> 마지막으로 Forwarding 단계에서 view 함수를 활용하여 이미지의 형태로 reshape 시켜주었습니다.


### 데이터셋 불러오기 및 전처리

<img width="849" alt="스크린샷 2022-08-10 오후 11 03 58" src="https://user-images.githubusercontent.com/52812351/183921322-9f1376ce-3657-41c9-b8e7-9118c023a526.png">

별도의 data augmentation은 적용하지 않았고 size를 확실히 해주기 위해서 Resize(28)만 적용해였습니다. 이후에는 텐서로 변환하였고, 기존에 0~1 사이였던 픽셀 값을 -1~1로 재설정했습니다.
배치 사이즈는 128로 설정했습니다.

### 모델 초기화 및 손실 함수, 학습률, 최적화 함수 설계

<img width="657" alt="스크린샷 2022-08-10 오후 11 14 08" src="https://user-images.githubusercontent.com/52812351/183923706-9a4fe13e-38f7-47f8-a595-bd29c43ac9dd.png">

생성자와 판별자 모두 모델 선언을 해주었고, gpu 할당을 해주었습니다.

Loss 함수로 BCELoss를 선언하였고, 학습률은 0.0002로 설정, 그리고 생성자와 판별자 모두 ADAM 옵티마이저로 최적화 시켜보았습니다.

### 모델 학습

<img width="667" alt="스크린샷 2022-08-10 오후 11 15 40" src="https://user-images.githubusercontent.com/52812351/183924072-e93387cb-73ef-44bd-aa02-592c8426e538.png">

우선 BCELoss에 target이 될 real과 fake를 각각 매번 batch size에 맞게 1.0과 0.0으로 선언했습니다.

> #### Generator 학습
> >Generator 먼저 학습을 시작하였고, 평균이 0 표준편차가 1인 정규분포에서 각 이미지마다 latent_dim 만큼의 random noise sampling을 진행하여 이를 Generator에게 입력값으로 전달했습니다.

Generator는 판별자가 Generator로 생성된 이미지에 대해 1에 가까운 값으로 평가하도록 학습을 진행해야 하기 때문에 

        g_loss = adverserial_loss(discriminator(generated_images), real)

를 통해 위와 같은 목적에 맞는 손실값을 설정했고, 이를 바탕으로 역전파 시키고, ADAM 옵티마이저를 통해 weigt update를 진행했습니다.

> #### Discriminator 학습
> > Generator는 진짜 이미지를 진짜라고 판단할 수 있는 능력, 그리고 가짜 이미지에 대해서는 가짜라고 판단할 수 있는 능력을 학습해야 하기 때문에 아래와 같은 방식으로 학습을 진행하였습니다.

<img width="602" alt="스크린샷 2022-08-10 오후 11 27 15" src="https://user-images.githubusercontent.com/52812351/183926925-8f0e9b97-f165-4e46-9d8c-109cc0055a50.png">

### 학습 결과
학습이 진행될수록 보다 선명하고 실제와 가까운듯한 MNIST 이미지들이 생성된걸 확인했습니다.

<img width="170" alt="스크린샷 2022-08-10 오후 11 28 21" src="https://user-images.githubusercontent.com/52812351/183927229-9fff55b8-82b4-4644-a58c-fe6bcc2ec946.png">
<img width="166" alt="스크린샷 2022-08-10 오후 11 28 44" src="https://user-images.githubusercontent.com/52812351/183927317-a632de5b-9701-4c73-acaf-7a8b2a354785.png">
<img width="169" alt="스크린샷 2022-08-10 오후 11 28 51" src="https://user-images.githubusercontent.com/52812351/183927349-17eb20fa-c010-4ab9-bab0-d9fea1f7c744.png">
<img width="164" alt="스크린샷 2022-08-10 오후 11 28 58" src="https://user-images.githubusercontent.com/52812351/183927373-24888b28-2516-433c-815f-416d27709cac.png">
<img width="164" alt="스크린샷 2022-08-10 오후 11 29 06" src="https://user-images.githubusercontent.com/52812351/183927412-a80d27ec-f530-45e7-a3e6-490cfed3ea37.png">





