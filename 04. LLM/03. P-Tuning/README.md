# P-Tuning

![image](https://github.com/user-attachments/assets/760ed0c4-74a3-4882-adbf-ad880561fe4b)

# P-Tuning이란?

P-Tuning은 대규모 언어 모델(LLM)을 미세 조정(Fine-Tuning)하는 방법 중 하나로, 학습 가능한 연속 벡터(Trainable Continuous Embedding)를 사용해 프롬프트를 최적화하는 방식.

- 기존 Prompt-Tuning은 사람이 직접 입력하는 고정된 텍스트 프롬프트를 사용하지만,
- P-Tuning은 모델이 학습할 수 있는 연속적인 임베딩 벡터(Continuous Prompt)를 추가하여 성능을 향상시킴.

즉, P-Tuning은 프롬프트를 사람이 직접 입력하는 대신 학습 가능한 파라미터로 변환한 것이라고 보면 된다.

## 1. P-Tuning이 필요한 이유

기존 LLM을 특정 태스크에 맞게 조정하려면 크게 두 가지 방법이 있다.

### (1) Full Fine-Tuning의 문제점
모델의 모든 가중치를 업데이트하면...
- 메모리 사용량이 너무 큼 (LLM은 수십~수백억 개의 파라미터를 가짐)
- 학습 속도가 느림
- 각 태스크마다 새로운 모델이 필요 (비효율적)

### (2) Prompt Engineering의 한계
사람이 직접 프롬프트를 입력하는 방식(Prompt Engineering)은...

- 프롬프트를 최적화하는 것이 어렵다
- LLM이 정해진 방식으로만 반응해야 해서 유연성이 낮음

해결책: P-Tuning은 학습 가능한 프롬프트를 도입하여, Full Fine-Tuning 없이도 성능을 높이는 방법

## 2. P-Tuning의 핵심 개념
P-Tuning은 입력 문장의 앞부분에 "Trainable Embedding"을 추가하는 방식으로 동작.

즉, 기존의 텍스트 프롬프트 대신, 모델이 학습할 수 있는 "연속 벡터(Continuous Prompt Embedding)"를 사용.

### (1) 기존 Prompt-Tuning 방식

Prompt-Tuning에서는 사람이 직접 텍스트 프롬프트를 입력.

예를 들어, 감성 분석(Sentiment Analysis) 모델을 학습할 때:

"This movie is amazing! The sentiment is [MASK]."
(라벨: Positive)

이 방식의 문제점은 프롬프트를 최적화하는 것이 어렵다.

### (2) P-Tuning 방식

P-Tuning에서는 학습 가능한 벡터를 프롬프트 앞에 추가하여 성능을 최적화.
즉, 텍스트가 아니라 연속적인 벡터(Trainable Embedding)를 사용해서 모델이 더 나은 성능을 내도록 학습.

``Output=Model([Trainable Prompt],Input)``

P-Tuning에서는 사람이 직접 프롬프트를 입력할 필요 없이, 모델이 최적의 프롬프트를 자동으로 학습할 수 있음.

## 3. P-Tuning vs Prefix-Tuning vs LoRA 비교

|방식|학습 대상|메모리 사용량|학습 속도|사용 방식|
|------|---|---|---|---|
|Full Fine-Tuning|모델 전체 가중치|많음|느림|새로운 모델 저장 필요|
|LoRA|특정 레이어의 일부 행렬|많음|빠름|Plug-and-Play 가능|
|Prefix-Tuning|Transformer 내부 Prefix 벡터|많음|빠름|Input에 Prefix 추가|
|P-Tuning|입력 프롬프트 벡터(Trainable Embedding)|매우 적음|매우 빠름|롬프트 최적화|

P-Tuning은 가장 적은 메모리로도 효과적인 Fine-Tuning이 가능

## 4. 예제

```
from transformers import AutoModelForCausalLM
from peft import PromptTuningConfig, get_peft_model

# 기본 LLM 모델 (예: GPT-2)
model = AutoModelForCausalLM.from_pretrained("gpt2")

# P-Tuning 설정
p_tuning_config = PromptTuningConfig(
    num_virtual_tokens=30,   # 학습할 프롬프트 길이
    task_type="CAUSAL_LM"    # 태스크 타입 (예: 언어 모델링)
)

# P-Tuning 모델 적용
p_tuning_model = get_peft_model(model, p_tuning_config)

# 학습 가능한 파라미터 개수 확인
p_tuning_model.print_trainable_parameters()

```

이렇게 하면 모델 전체를 학습하는 대신, "Trainable Prompt Embedding"만 학습하게 된다.


## 5. P-Tuning의 장점

1. Full Fine-Tuning보다 메모리 사용량이 훨씬 적음

대규모 모델(LLM)을 학습할 때 효율적

2. 텍스트 프롬프트보다 강력한 최적화 가능

Prompt Engineering보다 성능이 좋고, 모델이 직접 최적화 가능

3. 빠른 학습 & 다양한 태스크 적용 가능

감성 분석, 문장 요약, 번역 등 다양한 NLP 태스크에 사용 가능

4. 모델을 새로 저장하지 않아도 됨

LoRA처럼 Plug-and-Play 방식으로 활용 가능

## 6. 활용 사례

1) GPT, BERT 등의 성능 향상

기존 모델을 그대로 사용하면서도 최적의 프롬프트를 학습하여 성능을 향상

2) Zero-shot & Few-shot Learning 강화

적은 데이터로도 더 좋은 결과를 얻을 수 있음

3) RAG (Retrieval-Augmented Generation)에서 활용 가능

검색된 문맥을 최적화된 방식으로 모델에 입력 가능

4) 다양한 NLP 태스크 적용 가능

감성 분석, QA, 문서 요약, 챗봇 등 다양한 활용 가능

## 정리
- P-Tuning은 "학습 가능한 프롬프트 벡터(Continuous Prompt Embedding)"를 사용하여 LLM을 최적화하는 기법이다.
- 기존 Full Fine-Tuning보다 훨씬 적은 메모리로도 효과적인 성능을 낼 수 있다.
- Prompt Engineering보다 유연하고, 자동으로 최적화되는 방식이다.
- LoRA, Prefix-Tuning과 함께 Fine-Tuning을 대체할 수 있는 효율적인 방법이다.

특히 RAG + P-Tuning을 함께 활용하면, 검색된 문서를 최적의 형태로 모델에 입력하여 성능을 극대화할 수 있음!
