Vision Transformer with CIFAR 10
====================================================


## ViT의 탄생 배경
지금까지의 Attention은 Convolutional 연산과 함께 적용되거나 전체 구조를 유지하면서 CNN의 특정 구성 요소를 대체하는데 사용 되었습니다. 물론 Attention만을 이용한 모델들도 있었지만, ViT가 발표될 당시만 해도 CNN 모델을 뛰어넘지는 못했습니다. 그런데 ViT는 합성곱 연산을 전혀 사용하지 않고 이미지를 잘라서 patch 형태로 만든 다음에 Transformer에 패치들을 넣어서 이미지들을 분류 합니다.


## 장점
* Transformer의 구조를 거의 그대로 사용하기 때문에 확장성이 좋음
* Large scale 학습에서 매우 우수한 성능을 보임
* Transfer Learning시 CNN보다 훈련에 더 적은 computing resource를 사용

## 단점
Inductive bias의 부재로 CNN보다 학습을 위해 더 많은 데이터가 요구된다고 합니다.
예를 들어 중간 사이즈인 이미지넷을 강한 정규화 없이 학습에 사용할 경우, 유사한 크기의 ResNet보다 낮은 성능을 보인다고 합니다. 

기존 NLP: LSTM 위주의 Sequence 형식의 데이터를 받아와 처리하기 때문에 순서를 고려하여 예측을 할 수 있었지만 문장의 의미와 단어간의 관계가 순차적이지만은 않기 때문에 문장이 길어지게 되면 순서만으로 제대로된 번역을 할 수 없고 각 단어간의 관계를 알 수 없습니다. 또한 초반 sequence에서 결과물이 잘못 나왔다면 다음 sequence에도 영향을 미친다. 그래서 attention 기법을 활용해 전체 문장에 활용하려고 했던 것.


정보를 요청하는 것이 쿼리 -> 즉 주어진 데이터로 접근을 해서 유사성을 계산하는 것이기 때문에 key, value는 input에서, query는 label에서 불러오는것이 맞다.  


## Vision Transformer의 특징

#### Input
사진 한장이 통째로 들어간다.




### Multi-Head Attention
어텐션을 하나만 쓰는게 아니고 여러개 사용하여 학습에 따라서 어텐션들이 같은 것을 보더라도 다른 관점에서 정보를 처리할 수 있기 때문에 다수의 어텐션을 사용하여 그 결과를 조합하겠다는 의미.

### Layer Norm


### Masked Multi head attention
현재 위치를 기준으로 뒷부분은 어텐션이 적용되지 않도록 마스킹 적용.
