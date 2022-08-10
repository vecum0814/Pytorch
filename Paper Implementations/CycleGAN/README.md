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
Unpaired Dataset만을 가지고 있을 때 일반적인 vanilla GAN의 목표 함수만 사용한다고 생각해 봅시다.

