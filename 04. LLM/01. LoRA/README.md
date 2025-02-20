# LoRA

![image](https://github.com/user-attachments/assets/0428e129-b9c8-4dcb-9ecf-ebbe7594a93a)

# LoRA (Low-Rank Adaptation)란?

LoRA(Low-Rank Adaptation)는 대규모 언어 모델(LLM)이나 딥러닝 모델을 효율적으로 미세 조정(Fine-Tuning)하는 방법.
기존의 모든 가중치를 업데이트하는 대신, 일부 저차원(low-rank) 행렬만 학습하도록 제한해서 메모리와 연산량을 절약할 수 있음.

## 1. LoRA가 필요한 이유
### 기존 Fine-Tuning의 문제점
일반적으로 LLM을 Fine-Tuning할 때, 모든 모델 가중치를 업데이트해야 함.
하지만 대규모 모델(예: GPT-3, LLaMA, BERT)은 수십억 개의 파라미터를 가지고 있어서 메모리와 계산 비용이 엄청나게 증가함.

> 해결책: LoRA를 사용하면 모델의 모든 가중치를 학습하지 않고, 일부 저차원 행렬만 학습하여 성능을 유지하면서도 효율적으로 Fine-Tuning 가능

## 2. LoRA의 핵심 개념
LoRA는 기존 모델의 가중치(Weight)에 추가적인 저차원 행렬을 곱하는 방식으로 동작.

### (1) 기존 Fine-Tuning 방식
기존 방식에서는 모델의 모든 가중치 *W*를 업데이트함.

![image](https://github.com/user-attachments/assets/635bf10e-f5ad-4518-a4f7-91f1a65a322f)

- 여기서 *ΔW* 는 Fine-Tuning을 통해 학습되는 가중치 변화량.
- 문제점: 파라미터 수가 너무 많아서 메모리 사용량이 큼

### (2) LoRA 방식
LoRA는 직접 *ΔW*를 학습하지 않고, 두 개의 저차원 행렬 *A,B*의 곱으로 근사함.
![image](https://github.com/user-attachments/assets/e97376d4-1ea4-4257-9b79-2e47ca55fe9d)

여기서:

- A : d x r 행렬 ( Low-rank 행렬, 학습해야 할 값 )
- B : r x d 행렬 ( Low-rank 행렬, 학습해야 할 값 )
- r : Rank 값 ( 보통 4 ~ 16 정도로 설정 )

즉, 기존 파라미터를 직접 학습하는 대신, 작은 Rank r을 가진 행렬 A,B 만 학습.

**결과적으로 학습해야 할 파라미터 수가 훨씬 줄어들어 메모리와 연산 비용을 절약할 수 있음.**


## 3. LoRA의 장점
### 1. 메모리 절약
- 모델의 모든 가중치를 업데이트하지 않고, 작은 행렬 A,B만 학습 > VRAM 사용량 감소
### 2. 빠른 학습
- 학습할 파라미터 수가 적기 때문에 학습 속도가 훨씬 빠름
### 3. 원본 모델 유지 ( Plug-and-Play 가능 )
- LoRA를 사용하면 원본 모델을 변경하지 않고도 새로운 태스크에 맞춰 적응 가능
### 4. 여러 LoRA 가중치를 쉽게 교체 가능
- 하나의 LLM에 다양한 LoRA 모듈을 적용하여 다양한 태스크 전환 가능

## 4. LoRA 적용 방식
LoRA는 Transformer 모델의 특정 가중치(W_q, W_v)에만 적용하는 경우가 많음.

대부분 Self-Attention Layer의 Query(Q), Value(V) 행렬에만 LoRA를 적용해서 효율적으로 학습함.

## 5. 간단 예시

```
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# 기본 LLM 모델 (예: LLaMA)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# LoRA 설정
lora_config = LoraConfig(
    r=8,                    # Low-rank 차원 (작을수록 연산량이 적음)
    lora_alpha=16,          # Scaling factor
    lora_dropout=0.1,       # Dropout 적용
    bias="none",            # Bias 학습 여부
    target_modules=["q_proj", "v_proj"]  # LoRA를 적용할 레이어 선택 (주로 Q, V)
)

# LoRA 모델 적용
lora_model = get_peft_model(model, lora_config)

# 학습 가능한 파라미터 개수 확인
lora_model.print_trainable_parameters()

```
## 6. 사례
- LLM Fine-Tuning

GPT-3, LLaMA 같은 대형 모델을 효율적으로 미세 조정 가능

- 챗봇(Custom Chatbot)

특정 도메인(법률, 의료, 금융 등) 특화 챗봇을 만들 때 유용

- 멀티 태스크 모델 (Multi-task Model)

하나의 LLM에 여러 LoRA 어댑터를 적용하여 다양한 작업 수행 가능

- 경량 모델 배포

LoRA를 사용하면 전체 모델을 새로 학습하지 않고, 작은 LoRA 모듈만 로드하여 빠르게 배포 가능

## 정리

- LoRA(Low-Rank Adaptation)는 대형 모델을 효율적으로 Fine-Tuning하는 방법이다.
- 모델의 모든 가중치를 학습하지 않고, 저차원 행렬(LoRA Adapter)만 업데이트하여 연산량을 줄인다.
- 메모리 사용량이 적고, 속도가 빠르며, 원본 모델을 변경하지 않아도 Plug-and-Play 방식으로 활용할 수 있다.
- LLM Fine-Tuning, 챗봇, 경량 모델 배포 등 다양한 곳에서 유용하게 사용된다.

RAG + LoRA를 함께 활용하면 효율적으로 도메인 특화 LLM을 만들 수도 있다.
