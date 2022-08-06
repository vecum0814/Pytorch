Fashion MNIST Dataset with DNN and CNN
=======================================

## Fashion MNIST
Fashion MNIST 데이터셋이란 torchvision에 내장된 예제 데이터로, 운동화, 셔츠, 샌들 같은 작은 이미지의 모음이며, 기본 MNIST 데이터셋처럼 열 가지로 분류될 수 있는
28x28 픽셀의 이미지 7만개로 구성되어 있습니다.


# Cell 1 to Cell 4
필요한 라이브러리들을 불러왔으며, GPU 장치를 사용하기 위해 device를 설정해 주었습니다. 
torchvision에 내장된 torchvision.datasets.FashionMNIST를 통해 데이터셋을 불러왔고, 불러온 데이터에 대해 To.Tensor() Transform을 적용하여 넘파이 array를 torch.FloatTensor로 변환했습니다. 또한 [0, 255]이었던 픽셀의 범위를 [0.0, 1.0]으로 스케일링 해줍니다.

<img width="836" alt="스크린샷 2022-08-06 오후 6 11 59" src="https://user-images.githubusercontent.com/52812351/183242699-f5cd99a4-050e-4f7b-824b-66bc69fc466e.png">


# Cell 5-7
torch.utils.data.DataLoader를 사용해 batch_size = 100만큼 각각 train_dataset과 test_dataset을 불러왔습니다.
0-9까지 각 숫자 label마다 해당하는 의류의 이름을 명시해줬고, 이를 바탕으로 시각화하여 어떠한 데이터셋을 다루는지 알아보았습니다.
한 이미지에 대해서 size를 출력해 본 결과 [1, 28, 28]이 나왔고 이는 grayscale의 Height, Width가 28인 이미지를 의미합니다.

<img width="467" alt="스크린샷 2022-08-06 오후 6 11 23" src="https://user-images.githubusercontent.com/52812351/183242688-404b7808-8e3e-43f5-853f-5e4a0a58181c.png">

# Cell 8-9
Convolutional Neural Network와 Deep Neural Network의 비교를 위해 먼저 Convolutional Layer가 포함되지 않은 DNN 네트워크를 만들어 보았습니다.

<img width="551" alt="스크린샷 2022-08-06 오후 6 13 37" src="https://user-images.githubusercontent.com/52812351/183242746-35c84d5e-8b0e-405c-a505-bae02d82dfe3.png">

(28 x 28)인 각각의 의미지를 Linear Layer에 통과시켜 256의 output을 얻었고, Drop Out을 시켜줬습니다. 두 개의 Linear Layer를 더 커지면서 최종적으로 Fashion MNIST dataset의 label 종류인 10개의 텐서를 반환하게 설계했습니다.

[Batch_size, Channel, Height, Width]의 포캣으로 input_data가 들어오게 되는데, 이를 Linear Layer의 Input으로 주기 위해서는 텐서의 크기를 변경해 주어야 합니다.
이를 위해서 view함수를 사용하여 이미지의 Height와 Width에 해당하는 부분을 28 x 28 = 784로 크기를 변경해 주었습니다.

학습률은 0.001, loss 함수로는 Cross Entropy Loss를 선택했으며, optimizer로 Adam Optimizer를 선택했습니다.

# Cell 10
설정한 epoch만큼 모델을 train 했습니다. 
Train을 수행 하면서 test data가 들어있는 test_loader로부터 데이터를 불러와 학습이 잘 진행되고 있는지 확인하기 위해 모델이 예산한 값과 실제 값이 얼마나 차이가 나는지 
지속적으로 확인했으며, 500번의 iteration마다 Loss, Accuracy 값을 Print 했습니다.

<img width="581" alt="스크린샷 2022-08-07 오전 12 39 07" src="https://user-images.githubusercontent.com/52812351/183255911-fb7bf47f-ecf6-40a1-8cd2-1488b70833b0.png">

최종적으로 86%의 성능을 보였습니다.

# Cell 12-13
DNN 모델에 대한 성능을 평가해 보았기에, 이번엔 Convolutional Neural Network를 생성해 보았습니다.
Sequential 방식으로 (Conv Layer + 배치 정규화 + ReLU 활성화 함수 + 맥스 풀링) 이렇게 

두개의 레이어를 구성했고 Feature Extracting 부분이 종료되면 output으로 나온 Feature Map을 view 함수를 사용하여 Fully Connected Layer에 맞는 shape로 바꿔줬습니다.
그 이후로 FC 레이어를 통과하며, 최종적으로 10개의 텐서를 반환하도록 설계했습니다. 

<img width="588" alt="스크린샷 2022-08-07 오전 12 50 15" src="https://user-images.githubusercontent.com/52812351/183256280-c1c38871-c5b9-48c4-9599-07548097d599.png">

DNN의 경우와 동일하게 학습을 진행하였으며, 정확도는 약간 더 높았습니다.
