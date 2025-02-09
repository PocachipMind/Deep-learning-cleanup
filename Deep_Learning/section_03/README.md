
## 3-6 : 섹션4. [실습] PyTorch로 구현해보는 Loss Function

(17:30)

nn.BCELoss는 Ground Truth가 원핫 벡터 형태일 경우에 사용되는 Cross Entropy Loss 함수

nn.CrossEntropy는 Ground Truth가 Index Label 형태일 경우에 사용되는 Cross Entropy Loss 함수입니다.

두 함수 모두 동일한 수식으로 부터 비롯된 것입니다. 

두 함수의 차이는 오롯이 Ground Truth가 어떤 형태로 표현되느냐 그 차이뿐입니다.

( 18:30 )

시그모이드 레이어가 필요한 이유는 일반적으로 뉴런 네트워크의 마지막 레이어를 시그모이드 레이어로 두어

모델의 출력값이 0에서 1사이의 값으로 만들어 주기 위해서 입니다.
