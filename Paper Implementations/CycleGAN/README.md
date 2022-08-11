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

정리하자면, unpaired dataset을 가지고 adversarial loss만 사용하여 style transfer를 진행하면 target 도메인에 있을법한 이미지는 만들어 낼 수 있으나,
 




