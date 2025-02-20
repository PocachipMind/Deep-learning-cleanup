# Prefix-Tuning

![image](https://github.com/user-attachments/assets/b6347269-37ec-4b45-9dff-5f7b0a1e360e)

# Prefix-Tuning이란?

Prefix-Tuning은 대규모 언어 모델(LLM)을 효율적으로 Fine-Tuning하는 방법 중 하나.
기존 모델의 모든 가중치를 업데이트하는 대신, 입력 앞에 "Trainable Prefix"를 추가하여 모델을 조정하는 방식.

## 1. Prefix-Tuning이 필요한 이유
### 기존의 Fine-Tuning 방식은 모델의 모든 가중치를 업데이트해야 해서
- 메모리 사용량이 큼
- 학습 속도가 느림
- 여러 태스크를 지원하려면 모델을 여러 개 저장해야 함

### Prefix-Tuning은 모델의 기존 가중치는 그대로 두고, 추가적인 "Prefix 토큰"만 학습하기 때문에
- 훨씬 적은 메모리로 학습 가능
- Fine-Tuning보다 훨씬 빠름
- 하나의 모델로 여러 태스크 수행 가능


## 2. Prefix-Tuning의 핵심 개념

Prefix-Tuning은 입력 문장의 앞부분에 "Trainable Prefix"를 추가하는 방식으로 작동.
즉, **기존 모델의 파라미터는 Freeze(고정)** 하고,
추가된 Prefix 부분만 학습하여 원하는 태스크에 맞게 모델을 조정.


## 3. Prefix-Tuning이 동작하는 방식

Prefix-Tuning에서는 Transformer 모델의 각 층(레이어)에 학습 가능한 Prefix를 추가.
이 Prefix는 고정된 토큰이 아니라, 학습 가능한 벡터.

### (1) 기존 Transformer 구조
기본적으로 Transformer 모델(예: GPT, T5)은 입력 시퀀스를 인코딩한 후, 디코더에서 출력을 생성.

``Output=Decoder(Encoder(Input))``

### (2) Prefix-Tuning 방식
Prefix-Tuning에서는 입력 앞에 "Trainable Prefix"를 추가하여 모델이 다른 방식으로 동작하도록 유도.

``Output=Decoder(Encoder([Prefix,Input]))``

- Prefix 벡터는 학습 가능한 파라미터로, 모델이 새로운 태스크에 적응할 수 있도록 도움을 줌.
- 입력 문장은 그대로 유지되지만, Prefix 벡터가 문맥을 조정하는 역할을 함.
- Prefix는 고정된 토큰이 아니라, Continuous Embedding으로 학습됨.

## 4. Prefix-Tuning과 Prompt-Tuning 비교

|방식|학습 방식|추가 파라미터|적용 방법|
|------|---|---|---|
|Prompt-Tuning|단순한 텍스트 프롬프트 추가|없음|사람이 직접 입력|
|Prefix-Tuning|학습 가능한 벡터 추가|적음|모델이 Prefix를 학습|

- Prompt-Tuning은 사람이 직접 텍스트를 입력하는 방식
- Prefix-Tuning은 학습 가능한 벡터를 활용하는 방식

## 5. 간단 예시

```
from transformers import AutoModelForSeq2SeqLM
from peft import PrefixTuningConfig, get_peft_model

# 기본 LLM 모델 (예: T5)
model = AutoModelForSeq2SeqLM.from_pretrained("google/t5-small")

# Prefix-Tuning 설정
prefix_config = PrefixTuningConfig(
    num_virtual_tokens=20,   # Prefix 길이
    prefix_projection=True,  # Prefix를 Projection Layer로 변환
    task_type="SEQ_2_SEQ_LM" # 태스크 타입 (예: Seq2Seq)
)

# Prefix-Tuning 모델 적용
prefix_model = get_peft_model(model, prefix_config)

# 학습 가능한 파라미터 개수 확인
prefix_model.print_trainable_parameters()
```

이렇게 하면 원래 모델은 그대로 두고, Prefix 벡터만 학습하게 됨.

## 6. 사례

- 자연어 이해(NLU) 및 생성(NLG)

감성 분석, 문장 요약, 번역 등 다양한 태스크에 사용 가능

- LLM의 Few-shot Learning 강화

적은 데이터로도 새로운 태스크를 빠르게 학습 가능

- 여러 태스크를 하나의 모델에서 수행

하나의 LLM에 여러 개의 Prefix를 적용하여 멀티 태스크 수행 가능

- 경량 모델 배포

전체 모델을 저장하지 않고, Prefix 벡터만 저장하여 빠르게 배포 가능

## 정리
- Prefix-Tuning은 LLM을 효율적으로 Fine-Tuning하는 방법이다.
- 기존 모델의 모든 가중치를 학습하는 대신, "Trainable Prefix"만 학습하여 성능을 최적화한다.
- Full Fine-Tuning보다 메모리 사용량이 적고, 학습 속도가 빠르며, 여러 태스크를 하나의 모델에서 수행 가능하다.
- Prompt-Tuning과 다르게, Prefix-Tuning은 학습 가능한 벡터를 사용하여 보다 강력한 조정이 가능하다.
