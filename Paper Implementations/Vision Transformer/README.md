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
> > 학습을 진행할 때 한 배치당 각 클래스가 동일한 갯수로 들어올 수 있게 weighted random sampler를 도입했습니다. 

> PatchGenerator:
> > transform 안에서는 이미지 한장을 기준으로 진행해주기 때문에 (C, H, W)로 들어온다는 것을 인지했습니다. 
> > 이를 바탕으로 Height로 한번, Width로 한번 이미지를 썰어줬고, 올바른 shape로 구성해주었습니다.

### test.py

> accuracy:
> > 테스트에 필요한 함수를 구성해 주었습니다.

### vit.py
Argparse 함수를 사용하여 실험 환경을 조절할 수 있게 설계했습니다.

* D로 표현되는 latent_vec_dim을 argparse로 받았습니다.
* Transformer Encoder내에 존재하는 MLP의 노드 숫자를 임의로 int(latent_vec_dim / 2)로 설정했습니다.
* Patch 수는 논문에서 언급된 것과 같이 (H x W)/ P^2으로 설정해 주었습니다.

* Image Patch를 불러와야 하기 때문에 patchdata.py의 Flattened2DPatches를 통해 불러왔습니다.
* train, val, test loader 역시 불러와 주었고,
* 패치의 사이즈를 알기 위해 trainloader로부터 patch 덩어리 하나를 불러와봤습니다.

* model.py 함수에서 모델을 불러왔고, 이를 바탕으로 Train 및 Test를 진행했습니다. 


### model.py

> VisionTransformer:
> > 모델에 대한 전반적인 기능을 하는 함수로 설계하였고, 그 안에서 필요한 다른 함수들을 호출하는 방식입니다.
> > 들어온 데이터에 대해 Linear Projection을 수행해 줘야 하기 때문에 Linear Projection을 self.patchembedding이라는 이름으로 진행했습니다. 그 후 class token을 넣고 Patch embedding까지 진행했습니다.
> > Transformer Encoder가 Layer를 반복하기 때문에 그 반복을 list로 넣어봤습니다. 즉, TFencoderLayer를 num_layer만큼 list에 append합니다. 즉 각 class를 num_layer만큼 선언한 것이기 때문에 각 클래스는 parameter를 공유하지 않고 독립적인 parameter를 사용하게 됩니다. 
> > Transformer Encoder를 거친 값의 class token만 뽑아내서 classification을 할 것이기 때문에 mlp_head에서 이 부분에 맞게 Layer Norm과 Linear Layer를 통과하게 해줍니다.
> > Forward에선 각 이미지가 패치 임베딩을 거친 후 Transformer Encoder를 정해진 횟수만큼 거치며 mlp_head에 클래스 토큰만 전달하게 하였는데, 이 과정에서 attention 결과값도 받아볼 수 있게 설정했습니다.


> LinearProjection:
> > 1 x p^2 c -> 1 x D로 변환해 주는 것이기 때문에 차원 설정을 알맞게 해 주었습니다.
> > 클래스 토큰을 추가해주었고, Positional 파라미터까지 추가해줬습니다. 
> > linear projection 이후에 (B x N x D)가 되는 만큼, repeat 함수를 통해 cls_token의 사이즈를 (b x 1 x D)로 조절해 주었고
> > 포지셔널 임베딩을 더해 준 다음에
> > 드랍 아웃까지 추가해서 반환해 주었습니다.

> MultiheadedSelfAttention:
> > Linear Projection에서 나온 Input을 받습니다.
> > Query, Key, Value 모두 Linear로 정의해 주었는데, 원래 D를 D_h로 넘겨주어야 하지만 D_h == D / k인 만큼 D == k * D_h, 즉 D_h를 헤드 수 만큼 미리 계산 하는 방식으로 구현했습니다.
> > 각 Head마다의 Query, Key, Value를 구하기 위해서 num_head * head_dim == latent_vec_dim이라는 사실을 활용해 주었고 헤드 수를 앞으로 보내주기 위해 벡터의 갯수와 헤드 수의 위치를 바꿔주었습니다.
> > Query와 Key의 행렬곱에다가 softmax를 취하여 attention을 구하였고, 이를 value와 곱해주었습니다.
> > Cifar 10 데이터를 사용하고 실제 실험에 사용된 parameter들을 통해 진행할 경우, Attention의 size는 (100, 8, 257, 257)이 나오고 A x Value 는 (100, 8, 257, 4)가 나오게 됩니다.
> > 마지막에 Multi-Head를 concatenate 시키기 때문에 (100, 257, 32)로 나오게 됩니다. 

> TFencoderLayer
> > Transformer Encoder 하나에 대한 계산으로, 아래의 그림에서 필요한 연산들을 수행합니다.

<img width="236" alt="스크린샷 2022-08-16 오후 6 15 39" src="https://user-images.githubusercontent.com/52812351/184843900-a8de70a4-97c8-44f9-bf92-c49866b3a187.png">

