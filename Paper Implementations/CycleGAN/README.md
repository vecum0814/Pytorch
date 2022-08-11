Cycle GAN with Selfie2Anime dataset
====================================================

## Cycle GAN은 GAN을 기반으로 하는 Image to Image Translation을 제안한 논문입니다. Pairwise 되어 있지 않은 데이터셋에 대해서도 학습이 가능하다는 장점이 있습니다.

## CycleGAN의 탄생 배경
Pix2Pix를 사용하여 보다 general한 approach의 Image to Image Translation이 가능해졌지만, 한계 역시 명확했습니다. Pix2Pix는 서로 다른 두 도메인 X, Y의 데이터 두개를 하나의 pair로 묶어서 학습을 진행하는데요, colorization과 같이 이러한 데이터셋을 쉽게 구할 수 있는 task도 있겠지만 대부분의 경우 이러한 데이터셋을 구축하기 어렵습니다.

예를 들어 사진은 고흐가 그린 사진처럼 Image to Image Translation을 진행하려면 {사진, 그 사진을 고흐 풍으로 그린 그림} 이렇게 pair로 구성된 데이터가 적게는 몇 천장에서 많게는 몇 만장까지 필요한데, 이러한 데이터셋을 구축하는데는 너무나도 많은 시간과 비용이 필요합니다. 이러한 단점을 보완한 모델이 CycleGAN입니다.

## CycleGAN의 특징
CycleGAN은 우리가 비록 모네가 직접 그린 한라산을 보지 못했어도 모네의 그림들을 알고, 한라산이라는 풍경을 알 때 만약 모네라면 한라산을 이렇게 표현하지 않았을까? 하고 생각할 수 있는 것처럼 두 그룹의 stylistic difference를 인지하고 이에 따라 만약 이런 style을 translate 할 때 어떠한 결과물이 나올지 예측할 수 있는 네트워크 입니다. 

정리하자면, CycleGAN이란 paired training examples 없이도 특정한 image collection은 특별한 특징들을 잘 포착하여 이러한 특징들이 다른 image collection으로 translated 되었을 때 어떤 식으로 나올지에 대해 추론하는 모델입니다.

## 목표 함수
Unpaired Dataset만을 가지고 있을 때 일반적인 vanilla GAN의 목표 함수만 사용한다고 생각해 봅시다. 일반적인 GAN의 목표 함수를 사용하여 모델을 생성할 경우, 훈련이 잘 되었다는 가정 하에 X 도메인의 이미지를 매우 그럴싸한 Y 도메인의 이미지로 translate 시켜줄 수 있을 것입니다. 다만, 이러한 translation은 각각의 입력 x와 출력 y가 의미있게 paired up, 즉 이미지 x의 핵심 내용을 유지한채로 Y 도메인의 스타일로 translation이 되어 있다고 보장할 수 없습니다. Y 도메인에 대해 동일한 분포를 유도하는 G의 매핑이 무수히 많기 때문입니다. 또한, 'mode collapse' 즉 모든 input image들이 동일한 output image로 매핑이 되고 최적화 과정이 더 이상의 발전을 보이지 않는 현상이 자주 나타난다고 보고되었습니다.

다시 말해, GAN objective function만 사용 한다면 style transfer 과정에서 우리가 유지하고 싶은 x1의 content 정보를 아예 변경해 버릴 수 있기 때문에 추가적인 제약 조건이 필요합니다.

### Cycle Consistency
이러한 문제점을 해결하기 위해 CycleGAN에선 translation이 "cycle consistent"라는 속성을 가지게 하였습니다. 마치 어떠한 문장을 영어 -> 불어로 번역하고, 그 다음에 불어 -> 영어로 재번역 한다면 이 문장은 원 문장 그대로 돌아와야 하는게 바람직 할 것입니다. 수학적으로 표현하자면 우리에게 두 개의 trasnlator (X -> Y로 traslate 시키는 G, Y -> X로 trasnlate 시키는 F)가 있다면, G와 F는 역함수 관계여야 할 것이고, 매핑들 역시 1:1 대응이어야 할 것입니다. 논문에서는 이러한 구조적 가정을 mapping G와 F를 동시에 훈련시키고, F(G(x)) -> x와 G(F(y)) -> y를 encourage 시키는 cycle consistency loss를 추가함으로써 적용했습니다.

<img width="285" alt="스크린샷 2022-08-12 오전 12 22 38" src="https://user-images.githubusercontent.com/52812351/184169642-bf6c2728-3b36-45f6-b693-2f38792ad77b.png">


정리하자면, unpaired dataset을 가지고 adversarial loss만 사용하여 style transfer를 진행하면 target 도메인에 있을법한 이미지는 만들어 낼 수 있으나, 그렇게 나온 output image가 우리가 원하는 input image의 key feature를 가지고 있지 않을 확률이 높습니다. 그렇다면 output image에 input image의 key feature를 담는 방향으로 최대한 G를 훈련 시켜야 하는데, 이 과정에서 G(x)를 통해 생성된 y의 도메인인 Y에 속하는 이미지를 X 도메인으로 trasnlate 해주는 traslator F가 존재한다고 생각해 봅시다. 앞서 다뤘던 "cycle consistent"의 속성을 떠올려 본다면 F(G(x)) = x여야 합니다. 이 과정에서 F(G(x))의 output이 다시 x 그대로 나오게 하려면 G(x), 즉 y 자체에 x에 대한 key feature가 충분히 남아있어야 F 입장에서는 이러한 translation을 더욱 쉽게 적용할 수 있습니다. 

<img width="208" alt="스크린샷 2022-08-12 오전 12 22 56" src="https://user-images.githubusercontent.com/52812351/184169708-9f640fa5-6e0a-438a-aac0-6e44d9ccab45.png">

이렇게 G가 X 도메인 이미지들의 key feature들을 최대한 보존하면서 Y 도메인의 이미지들을 생성하는 방식으로 학습한다면, 최종적으로 G는 input image의 key feature는 최대한 유지하면서 우리가 원하는 도메인의 이미지로 image translation을 적용할 수 있게 됩니다.
 

최종적으로 위에서 언급했던것과 같이 논문에서는 objective function에 mapping G와 F를 각각의 Discriminator에 대해 동시에 훈련시키고, F(G(x)) -> x와 G(F(y)) -> y를 encourage 시키는 cycle consistency loss를 추가함으로써 unpaired data에서도 image의 key feature는 유지하며 style만 바꾸는 방법이 작동하도록 설계했습니다.

 <img width="996" alt="스크린샷 2022-08-12 오전 12 28 40" src="https://user-images.githubusercontent.com/52812351/184170997-d601c22f-6a5d-41d4-8dcb-0c68deebc52f.png">


<img width="696" alt="스크린샷 2022-08-12 오전 12 27 42" src="https://user-images.githubusercontent.com/52812351/184170780-c1e9d833-c4c3-4734-a20b-d1ccf29f4e36.png">

## 네트워크 아키텍쳐
* Residual Block을 활용하는 아키텍쳐이고, instancen normalization을 활용합니다.
* Pix2Pix와 마찬가지로 Discriminator에서 PatchGAN을 활용하여 이미지의 진위 여부를 판별합니다.

## Training details
* 기존의 cross-entropy 기반의 Loss 대신에 MSE 기반의 loss를 사용합니다. 이를 사용하여 실제 이미지 분포와 더욱 가까운 이미지를 생성할 수 있었고, 학습이 안정화되었습니다.
* Model oscillation을 개선하기 위해 가장 최근에 Generator로부터 생성된 이미지 대신 생성자가 이전에 만든 50개의 이미지를 저장해두고, 이를 활용하여 Discriminator를 업데이트 합니다.

## 실험 결과

<img width="640" alt="스크린샷 2022-08-12 오전 1 16 52" src="https://user-images.githubusercontent.com/52812351/184180793-40af45f5-5e8d-4df9-b787-e71fe7f85161.png">

Paired dataset을 이용해 학습한 Pix2Pix와 비교할 만한 결과가 나왔습니다.


## Selfie2Anime dataset을 활용한 CycleGAN 실습

### Residual Block
Generator를 구현하기 위해 내부에 들어갈 Residual Block을 구현했습니다. Residual Block은 이전 layer와 현재 layer의 출력값을 더해서 Forward하기 때문에 모델이 깊어짐에 따라 생기는 기울시 소실 문제를 해결합니다. 

<img width="412" alt="스크린샷 2022-08-12 오전 1 22 53" src="https://user-images.githubusercontent.com/52812351/184181870-332e1c31-cdcc-4442-a098-c3e0d324176b.png">

### Generator

<img width="640" alt="스크린샷 2022-08-12 오전 1 27 13" src="https://user-images.githubusercontent.com/52812351/184182679-610acf78-1e13-45ac-af29-a8da898fe4d4.png">


입력 이미지의 Height와 Width를 2배씩 줄여주는 다운 샘플링을 진행한 후 여러개의 Residual Block을 통가시킨 후에 다시 Height와 Width를 2배씩 늘려주는 Upsampling을 진행하는 방식으로 스타일을 변환하는 Generator를 구현했습니다.

### Discriminator 

<img width="671" alt="스크린샷 2022-08-12 오전 1 29 22" src="https://user-images.githubusercontent.com/52812351/184183086-c6f453aa-02a7-466e-996c-53bcfe2771c2.png">

차원의 크기는 지속적으로 늘리되 stride = 2로 설정하여 이미지의 크기는 줄이는 Downsampling을 진행하여 이미지의 크기를 지속적으로 줄여주었고, 마지막에는 PatchGAN 형식의 Discriminator인 만큼 적절한 사이즈로 설정하여 반환했습니다.

### 모델 초기화 및 Hyperparameter 설계

<img width="443" alt="스크린샷 2022-08-12 오전 1 31 57" src="https://user-images.githubusercontent.com/52812351/184183573-fec3732d-18ca-47aa-bd41-9fd113ea1a5e.png">
<img width="659" alt="스크린샷 2022-08-12 오전 1 32 26" src="https://user-images.githubusercontent.com/52812351/184183667-16679a2d-0758-4fa0-abcc-0ac59ab6b940.png">

논문에서 언급한 방식대로 가중치 초기화를 진행해 주었고, 


