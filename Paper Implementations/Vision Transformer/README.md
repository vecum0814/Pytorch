Vision Transformer with CIFAR 10
====================================================


## ViT의 탄생 배경
지금까지의 Attention은 Convolutional 연산과 함께 적용되거나 전체 구조를 유지하면서 CNN의 특정 구성 요소를 대체하는데 사용 되었습니다. 물론 Attention만을 이용한 모델들도 있었지만, ViT가 발표될 당시만 해도 CNN 모델을 뛰어넘지는 못했습니다. 그런데 ViT는 합성곱 연산을 전혀 사용하지 않고 이미지를 잘라서 patch 형태로 만든 다음에 Transformer에 패치들을 넣어서 이미지들을 분류 합니다.

Transformer가 먼저 적용된 NLP 분야에서 꾸준히 사용되어왔던 LSTM 위주의 Sequence 형식의 데이터를 받아와 처리하는 기법은 순서를 고려하여 예측을 할 수 있었지만, 문장의 의미와 단어간의 관계가 순차적이지만ㅇ느 않기 때문에 문장이 길어지게 되면 순서만으로는 제대로 된 번역을 할 수 없었고, 각 단어간의 관계들도 알 수 없었습니다. 또한 초반 sequence에서 결과물이 잘못 나왔다면, 다음 sequence에도 영향을 미쳤기 때문에 Attention 기법을 활용해 전체 문장에 활용하려 했습니다.

## 장점
* Transformer의 구조를 거의 그대로 사용하기 때문에 확장성이 좋음
* Large scale 학습에서 매우 우수한 성능을 보임
* Transfer Learning시 CNN보다 훈련에 더 적은 computing resource를 사용

## 단점
Inductive bias의 부재로 CNN보다 학습을 위해 더 많은 데이터가 요구된다고 합니다.
예를 들어 중간 사이즈인 이미지넷을 강한 정규화 없이 학습에 사용할 경우, 유사한 크기의 ResNet보다 낮은 성능을 보인다고 합니다. 


정보를 요청하는 것이 쿼리 -> 즉 주어진 데이터로 접근을 해서 유사성을 계산하는 것이기 때문에 key, value는 input에서, query는 label에서 불러오는것이 맞다.  


## Vision Transformer의 특징

<img width="442" alt="스크린샷 2022-08-16 오후 4 54 32" src="https://user-images.githubusercontent.com/52812351/184827540-e609cb4a-e925-4816-874d-6559e933819a.png">


#### Input
> NLP에서의 Transformer가 단어별로 나뉜 문장 전체를 입력받는 것 처럼, Vision Transformer에서는 이미지 하나를 정해진 크기의 Patch들로 나눈 이후에, Transformer Encoder로 들어가게 됩니다.
> Transformer Encoder에 들어가기 전에 Patch들을 Flatten 시켜 Linear Layer에 통과시켜 원하는 Embedding Dimension만큼의 차원을 가지게 한 후, 최종적으로 Classification에 사용될 class token을 각 Patches에 추가해주고, Position Embedding을 행렬 덧셈 연산 시켜주면서 이미지의 위치 정보까지 추가해 줍니다. 


### Multi-Head Attention
> Transformer, 그리고 Vision Transformer에서 사용된 어텐션 기법은 Database에서 Query가 주어지면 그에 맞는 key 값을 찾아서 data를 가져오는 것처럼 Key & Query간의 유사도를 구하는 행위입니다. Query와 Key값을 행렬곱 시켜주어 그 둘의 유사도를 구하고, 이것을 softmax 활성화 함수에 통과시켜서 어떤 부분에 가중치를 두어 볼지를 정하는 Attention Map을 생성합니다. 이를 Value에다가 곱함으로써, Value에서 어떤 부분이 중요한지 나타낼 수 있습니다.
> Multi-Head Attention이란 하나의 어텐션만을 사용하는 것이 아니라 여러개 사용합니다. 학습에 따라서 어텐션들이 같은 것을 보더라도 다른 관점에서 정보를 처리할 수 있기 때문에 다수의 어텐션을 활용하여 그 결과를 조합합니다. 
어텐션을 하나만 쓰는게 아니고 여러개 사용하여 학습에 따라서 어텐션들이 같은 것을 보더라도 다른 관점에서 정보를 처리할 수 있기 때문에 다수의 어텐션을 사용하여 그 결과를 조합하겠다는 의미.

### Layer Norm
> 각 feature, 즉 D차원에 대해서 정규화를 진행해 줍니다. 


## Transformer Encoder 구조

<img width="111" alt="스크린샷 2022-08-16 오후 4 56 43" src="https://user-images.githubusercontent.com/52812351/184827933-1b5d6967-d987-4281-b043-33d25a8b912b.png">

* 우선 Patch + Positional Embedding까지 끝난 Embedded Patches를 입력으로 받아옵니다.
* Layer Normalization을 거치고
* Query, Value, Key를 통해 Activation을 구하고
* Activation을 Value와 곱해서 Self Atention을 구합니다. 
* Multi head Attention들을 Linear에 한번 통과시켜주고
* Skip Connection을 통해서 첫번째 Layer Normalization을 거치지 않은 정보와 더해줍니다.
* Layer Normalization을 한번 더 거치고
* 이를 MLP에 통과시켜준 다음 (FC층 2개와 GELU 활성화 함수 사용)
* 두번째 Layer Normalization 직전의 정보와 한번 더 Skip connection을 통해 더해주고
* 이러한 전체 과정을 L번 반복합니다.

## MLP Head

<img width="253" alt="스크린샷 2022-08-16 오후 5 03 13" src="https://user-images.githubusercontent.com/52812351/184829148-81f62f3e-9bac-4a0c-99d2-c5839454bda3.png">

(N + 1) x D의 shape를 가지고 있는 입력에서 클래스 토큰 부분만 분류에 사용합니다. Layer Normalization을 적용하고 Fuly Connected Layer를 거쳐 prediction 값을 생성합니다.

## 실험 결과

<img width="495" alt="스크린샷 2022-08-16 오후 5 05 31" src="https://user-images.githubusercontent.com/52812351/184829643-14f938cb-fa5a-4690-96cd-7396284e0855.png">

N x (p^2 c) shape를 Linear Projection을 거쳐 N x D shape로 만드는 Embedding Filter의 첫 28개의 속성을 나타낸 그림입니다. 이미지 분류를 수행하는 CNN의 초기 Layer들에서 볼 수 있는 국소적 특징들을 다루는 filter라고 판단 할 수 있습니다.

<img width="363" alt="스크린샷 2022-08-16 오후 5 07 41" src="https://user-images.githubusercontent.com/52812351/184830059-0f089858-41a8-452d-883b-fbdff65fbe34.png">

패치들간의 유사도를 보여주는 그림입니다. 상대적으로 가까운 거리의 패치일수록 높은 cosine similarity를 보이고, 반대일 경우 낮은 유사도를 보인다는 것으로 보아 위치 정보가 잘 학습되었다고 할 수 있습니다.

<img width="332" alt="스크린샷 2022-08-16 오후 5 09 22" src="https://user-images.githubusercontent.com/52812351/184830363-19436f7d-36e1-4bf8-a059-00035ba6533a.png">

네트워크의 깊이와 그에 따른 평균 Attention distance를 보여주는 그래프입니다. Attention Map은 Query와 Key의 행렬곱 연산을 수행한 뒤에 Softmax 연산을 통해 구해지게 되는데, 이 때 얼마나 가까운 거리, 혹은 먼 거리의 픽셀들까지 의미있게 고려했나에 따라서 Mean attention distance가 결정되게 됩니다. Network Depth가 얕을때는 가까운 거리 위주로 집중하는 추세가 보였다면, 네트워크가 깊게 들어갈 수록 전체를 보는 경향이 두드러지는데, 이러한 특징은 CNN에서도 Convolutional Layer가 깊어질수록 이전 Convolutional Layer의 output을 바탕으로 더욱 넓은 관점에서 보는 것과 비슷하다고 볼 수 있겠습니다.


## Vision Transformer with CIFAR 10

### patchdata.py
Original data를 Patch data로 변환하는 함수입니다.

> Flattened2dPatches:
> > 입력으로 주어지는 img_size, patch_size, batch_size에 따라 data를 생성할 수 있도록 구현했습니다. 
> > data name에 따라 데이텃세에 맞는 평균과 표준편차를 이용해 정규화를 진행할 수 있게 하였고 dataset을 적당한 transforms과 함께 구성했습니다. 이 때 PatchGenerator라는 함수는 직접 만들어 적용시켜 주었습니다.
> > 학
> > data name에 따라 데이텃세에 맞는 평균과 표준편차를 이용해 정규화를 진행할 수 있게 하였고 dataset을 적당한 transforms과 함께 구성했습니다. 이 때 PatchGenerator라는 함수는 직접 만들어 적용시켜 주었습니다.

> PatchGenerator:
> > transform 안에서는 이미지 한장을 기준으로 진행해주기 때문에 (C, H, W)로 들어온다는 것을 인지했습니다. 
> > 이를 바탕으로 Height로 한번, Width로 한번 이미지를 썰어줬고, 올바른 shape로 구성해주었습니다.




