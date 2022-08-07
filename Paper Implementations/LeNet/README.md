LeNet with Cat vs dog classification problem
====================================================

## LeNet은 합성곱 신경망이라는 개념을 최초로 얀 르쿤 교수가 개발한 구조입니다. 합성곱 연산관 풀링을 반복적으로 거치면서 마지막에 완전 연결층에서 분류를 수행합니다.


## LeNet의 탄생 배경
기존의 Fully Connected Layer만을 사용하는 모델로도 이미지 분석이 가능하긴 하였지만, 여기에는 몇가지의 문제점들이 존재했습니다.
<img width="468" alt="스크린샷 2022-08-07 오전 1 15 57" src="https://user-images.githubusercontent.com/52812351/183257254-57f23ba5-c007-40f8-9cae-87985c2d9c34.png">

  > 위치 정보 손실
  일반적인 Linear Layer만을 사용하여 구성된 DNN의 경우, 이미지 정보를 입력으로 받을 때 Linear Layer에 넣기 위해 2차원인 (H, W)데이터를 1차원으로 바꿉니다.
  이 과정에서 실제 위치 정보가 누락되게 되는데 그러다보니 글자 이미지의 위치가 조금만 달라지거나 회전, 또는 변형이 조금만 생겨도 모델 입장에서는 다른 이미지라고 판단할 수 있기 때문에 변형된 이미지로
  새롭게 학습해야 합니다. 이처럼 많은 데이터로 학습시켜야 할 필요성이 있기 때문에 훈련에 많은 시간이 소요됩니다. 
  
  > 너무 많은 Parameter 수
  16x16 사이즈의 필기체를 인식하기 위해 사진과 같은 간단한 모델에서도 학습을 위한 Parameter의 개수가 약 3만개에 달합니다. 만약 이미지의 크기가 더 커지고, 흑백 이미지가 아닌 컬러 이미지가 사용되고, 
  레이어를 더 깊이 쌓을 수록 Parameter 크기는 기하급수적으로 늘어납니다. 이처럼 Parameter수가 많아짐에 따라, 학습 시간도 오래 걸릴 뿐더러 과적합 문제도 발생할 가능성이 커집니다.
  
 ## CNN의 특징
 
  > Receptive Field
  1958년에 진행된 고양이 실험을 통해 시각 피질 안의 많은 뉴런이 작은 Local Receptive Field를 가진다는 것을 밝혀졌으며, 이는 뉴런들이 시야의 일부 범위 안에 있는 시각 자극에만 반응 한다는
  의미입니다. 이러한 Receptived Filed들은 서로 겹칠 수 있으며, 이렇게 겹쳐진 Receptive Field들이 전체 시야를 이루게 됩니다. 추가적으로 어떤 뉴런은 수직선의 이미지에만 반응하고,
  다른 뉴런은 다른 각도의 선에 반응하는 뉴런이 있을 뿐만 아니라, 어떤 뉴런은 큰 Receptive Field를 가져 저수준의 패턴(edge, blob)이 조합되어 복잡한 패턴 (texture, object)에 
  반응한다는 것을 알게 되었습니다. 이러한 관찰을 통해 고수준의 뉴런이 이웃한 저수준의 뉴런의 출력에 기반한다는 아이디어를 생각해 냈습니다.
  
  > Translation Invariant
  이미지의 특정한 부분이 다른 위치에서 동일하게 등장한다면, 커널을 통해 합성곱 연산이 진행된 결과물은 서로 일치하게 됩니다. 즉, 위치가 달라도 동일한 feature를 추출할 수 있는 장점이 있습니다.
  
  > Local Relationship
  합성곱 연산의 결과물은 Feature Map은 locally connected되어 있기 때문에, 이미지의 공간 정보를 유지하면서 인접 이미지의 특징을 효과적으로 인식합니다.
  
  ## Pooling Layer
  Pooling(Subsampling)레이어는 Conv Layer를 통과한 Feature Map을 입력으로 받아서 Feature Map의 크기를 줄이며 특정 데이터를 강조하는 용도로 사용됩니다. 
  다양한 종류들의 Pooling 방법이 존재하는데, 사용자의 판단에 따라 Pooling을 진행하려는 영역을 대표하는 값이 어떤 값일지에 따라 다양하게 사용할 수 있습니다. 
  즉, Feature Map의 영역 중에서 그 영역을 대표할 특징을 선별하는 과정이며, 이 과정에서 Feature Map의 크기를 줄여줍니다.
  
  Feature Map의 크기를 줄여주는 이유는, 앞선 Layer들을 거치고 나서 나온 Output Feature Map의 모든 데이터가 필요하지 않기 때문입니다.
  다시말해, 추론을 하는데 있어 적당량의 데이터만 있어도 충분하기 때문입니다. 
  
  Pooling을 진행할 경우 쓸데없는 parameter의 수를 줄일 수 있으며, 과적합을 줄일 수 있습니다.
  
  
  <img width="655" alt="스크린샷 2022-08-07 오전 2 32 58" src="https://user-images.githubusercontent.com/52812351/183259703-1359bcdc-11a6-4edd-b2c3-b604d654c672.png">

## 모델 구조에 대해

<img width="675" alt="스크린샷 2022-08-07 오전 12 57 51" src="https://user-images.githubusercontent.com/52812351/183256549-2c0dbb0a-3a7f-49fe-b16d-2171b1784f8c.png">

32 x 32의 입력이 들어올 때, C1에서 5x5 size의 커널을 통해 합성곱 연산을 진행한 후 28 x 28 크기의 Feature Map을 5개 생성합니다.
S2에서 다운 샘플링하여 특성 맵 크기를 14 x 14로 줄이고, 다시 C3에서 5x5 합성곱 연산을 진행하여 10 x 10 크기의 Feature Map을 16개 생성합니다.
S4에서 다운 샘플링하여 특성 맵 크기를 5x5로 줄이고 C5에서 5x5 합성곱 연산을 수행하여 1x1 크기의 Feature Map을 120개 생성합니다.
마지막으로 F6에서 완전연결층으로 C5의 결과를 Node 84개에 연결 시키고 10개의 Node를 가진 출력을 리턴합니다.



# Cat VS Dog 데이터로 LeNet 구현

## 이미지 데이터셋 전처리

<img width="828" alt="스크린샷 2022-08-07 오전 3 01 09" src="https://user-images.githubusercontent.com/52812351/183260577-9861e369-e412-44fb-9d6b-2e57fbd8b62b.png">

이미지 전처리를 진행할 수 있는 ImageTransform() 클래스를 생성하였습니다.
train을 진행할 때는 RandomResizedCrop, RandomHorizontalFlip, ToTensor, Normalize를 진행하여 과적합을 줄이며 dataset을 늘리는 효과를 가져오는 data augmentation 기법들을
수행했으며, 이를 텐서로 바꿔주고 정규화를 진행하도록 하였습니다.
test를 진행할 때는 이미지 Resizing, CenterCrop과 같은 약한 augmentation만 적용해주고 이를 텐서로 바꾸고 정규화를 진행하도록 하였습니다.

<img width="551" alt="스크린샷 2022-08-07 오전 3 06 01" src="https://user-images.githubusercontent.com/52812351/183260759-11cc3f0d-714b-4694-a521-cfe7d3cefa89.png">

torch.utils.data의 Dataset을 상속받아 Custom Dataset을 만들었고, Imagenet에서 사용된 mean과 std를 transform에 적용하였습니다.

## DataLoader

<img width="685" alt="스크린샷 2022-08-07 오전 3 08 38" src="https://user-images.githubusercontent.com/52812351/183260827-6cb584c0-a91a-4f89-b9f3-a3c97863e472.png">


torch.utils.data의 DataLoader를 사용하여 train_dataloader와 val_dataloader를 설정했습니다. 
batch_iterator를 통해 불러온 input의 사이즈는 [32, 3, 224, 224]로, 의도한대로 잘 나온것을 확인할 수 있었습니다.


## 모델의 네트워크 클래스 정의

<img width="850" alt="스크린샷 2022-08-07 오전 3 13 43" src="https://user-images.githubusercontent.com/52812351/183261011-40d2388b-abd1-4591-a251-95ad27686071.png">

기존의 LeNet 모델에서는 첫번째 Conv Layer에서 6개의 filter를 사용했고 두번째 Conv Layer에는 16개의 filter를 사용하였습니다.
하지만 이때 사용된 입력 이미지가 1채널로만 구성된 이미지라는 부분을 고려해서 현재 데이터셋은 3채널로 구성되어 있는 만큼 Conv Layer의 filter 수를 original 논문보다 늘려서 
각 filter가 더욱 다양한 특성을 뽑아낼 수 있도록 의도해 보았습니다.

마지막에는 Softmax 활성화 함수를 사용하여 [0, 1]의 범위로 예측한 결과를 받아서 0.5보다 클 경우 강아지, 0.5 이하일 경우 고양이로 분류하도록 설계했습니다.


## 옵티마이저와 손실 함수 정의

<img width="555" alt="스크린샷 2022-08-07 오후 4 29 23" src="https://user-images.githubusercontent.com/52812351/183280217-b803c3ca-9cde-4146-a82e-861bfb362e8d.png">

원 논문에서는 옵티마이저로 전체 샘플에서 하나의 샘플만 추출하고 그것의 기울기만 반영하는 SGD를 사용하였지만 이번 구현에서는 모멘텀 SGD를 사용했습니다. 모멘텀 SGD는 SGD에 관성이 추가된 것으로, 매번 기울기를 구하지만 가중치를 수정하기 전에 이전 수정 방향을 참고하여 같은 방향으로 일정한 비율만 수정되게 하는 방법입니다. 손실 함수는 크로스 엔트로피를 사용했습니다.

## 모델 학습 함수 정의

<img width="746" alt="스크린샷 2022-08-07 오후 4 35 52" src="https://user-images.githubusercontent.com/52812351/183280438-6fab7dff-7bfd-4950-ae20-674fbf86dfc6.png">

모델 학습에 걸렸던 시간을 추적할 수 있게 설계했고, 매 epoch마다 train과 valid를 번갈아 수행하며 해당 epoch에서 발생한 loss와 총 정답 갯수를 바탕으로 구한 정확도를 표기하였습니다.
실행하고 있는 phase가 'train'일 때만 torch.set_grad_enabled(phase == 'train')을 통해 기울기 연산이 활성화 되게 설정했습니다.

## 모델 학습

<img width="610" alt="스크린샷 2022-08-07 오후 4 42 55" src="https://user-images.githubusercontent.com/52812351/183280681-5c5a90f7-5c95-4e07-800c-aad5ce47cb50.png">

총 30회의 epoch동안 학습을 진행하였고, 결과는 다음과 같았습니다.

<img width="242" alt="스크린샷 2022-08-07 오후 4 43 29" src="https://user-images.githubusercontent.com/52812351/183280699-a894d88a-dbeb-439e-ac8d-d1f8eab273d9.png">


