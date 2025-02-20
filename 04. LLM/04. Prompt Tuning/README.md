# Prompt Tuning

![image](https://github.com/user-attachments/assets/434e27ed-1351-4b97-8593-8a1ba59eddb5)

런에이블한 임베딩 벡터를 어떻게 초기 이니셜라이제이션 할 것인지, 임베딩 벡터 개수를 몇개로 할 것인지.

프롬프트 튜닝의 디자인 이슈 ( 하이퍼 파라미터 )

PreFix Tuning은 이 앞에 붙여주는 거를 모든 트랜스포머 레이어에 다 넣어주는 것 프롬프트 튜닝은 처음 인풋 x 앞에 한번만 넣어준다. 

P Tuning은 앞뒤로 넣기 때문에 좀 더 복잡한 구조. 그리고 파라미터도 LSTM 학습을 하기에 프롬프트 튜닝 비해서 학습 파라미터가 더 많을 수 있다.

## Prompt Tuning이란?

Prompt Tuning은 대규모 언어 모델(LLM)을 효율적으로 미세 조정(Fine-Tuning)하는 방법 중 하나.
기존의 Fine-Tuning 방식과 달리, 모델의 가중치는 그대로 유지하고, 학습 가능한 "프롬프트(Trainable Prompt Embedding)"를 추가하여 성능을 조정하는 방식.

즉, 모델을 직접 업데이트하는 대신, 프롬프트를 최적화하는 방식.

## 1. Prompt Tuning이 필요한 이유

기존 LLM을 특정 태스크에 맞게 학습하는 방식에는 몇 가지 문제점이 있음.

### (1) Full Fine-Tuning의 한계
- 모델의 모든 가중치를 업데이트해야 하므로 학습 시간이 오래 걸림.
- GPU 메모리를 많이 사용해서 비용이 많이 듬.
- 각 태스크마다 새로운 모델을 저장해야 해서 비효율적.
### (2) Prompt Engineering의 한계
- 사람이 직접 텍스트 프롬프트를 입력해야 함.
- 프롬프트를 최적화하는 것이 어렵고, 일반화가 잘 안 됨.

해결책: Prompt Tuning은 학습 가능한 프롬프트를 도입하여, Fine-Tuning 없이 성능을 향상하는 방법

## 2. Prompt Tuning의 핵심 개념
Prompt Tuning은 입력 문장의 앞부분에 "Trainable Prompt Embedding"을 추가하는 방식으로 동작.

즉, 기존의 텍스트 프롬프트 대신, 모델이 학습할 수 있는 "연속적인 벡터(Continuous Prompt)"를 사용하는 것.

### (1) 기존 Prompt Engineering 방식
Prompt Engineering에서는 사람이 직접 텍스트 프롬프트를 입력해.
예를 들어, 감성 분석(Sentiment Analysis) 모델을 학습할 때:

"This movie is amazing! The sentiment is [MASK]."
(라벨: Positive)


이 방식의 문제점은 프롬프트를 최적화하는 것이 어렵고, 일반화가 잘 안 된다는 점.

### (2) Prompt Tuning 방식

Prompt Tuning에서는 학습 가능한 벡터를 프롬프트 앞에 추가하여 성능을 최적화.

즉, 텍스트가 아니라, 연속적인 벡터(Trainable Embedding)를 추가하여 모델이 더 나은 성능을 내도록 학습하는 방식.

Output=Model([Trainable Prompt],Input)

Prompt Tuning에서는 사람이 직접 프롬프트를 입력할 필요 없이, 모델이 최적의 프롬프트를 자동으로 학습할 수 있음.

## 3. Prompt Tuning vs P-Tuning vs Prefix-Tuning 비교

|방식|학습 대상|메모리 사용량|학습 속도|사용 방식|
|------|---|---|---|---|
|Full Fine-Tuning|모델 전체 가중치|많음|느림|새로운 모델 저장 필요|
|Prefix-Tuning|Transformer 내부 Prefix 벡터|적음|빠름|Input에 Prefix 추가|
|P-Tuning|입력 프롬프트 벡터(Trainable Embedding)|매우 적음|매우 빠름|프롬프트 최적화|
|Prompt Tuning|입력 앞에 추가되는 학습 가능한 벡터|적음|빠름|특정 태스크에 최적화된 프롬프트 생성|

Prompt Tuning은 P-Tuning과 유사하지만, 특정 태스크에 맞게 최적화된 연속 벡터를 학습하는 것이 차이점

## 4. 예제

```
from transformers import AutoModelForSeq2SeqLM
from peft import PromptTuningConfig, get_peft_model

# 기본 LLM 모델 (예: T5)
model = AutoModelForSeq2SeqLM.from_pretrained("google/t5-small")

# Prompt Tuning 설정
prompt_config = PromptTuningConfig(
    num_virtual_tokens=20,   # 학습할 프롬프트 길이
    task_type="SEQ_2_SEQ_LM" # 태스크 타입 (예: Seq2Seq)
)

# Prompt Tuning 모델 적용
prompt_model = get_peft_model(model, prompt_config)

# 학습 가능한 파라미터 개수 확인
prompt_model.print_trainable_parameters()

```

이렇게 하면 모델 전체를 학습하는 대신, "Trainable Prompt Embedding"만 학습하게 됨.

## 5. Prompt Tuning의 장점

1. Full Fine-Tuning보다 메모리 사용량이 훨씬 적음

대규모 모델(LLM)을 학습할 때 효율적

2. 텍스트 프롬프트보다 강력한 최적화 가능

Prompt Engineering보다 성능이 좋고, 모델이 직접 최적화 가능

3. 빠른 학습 & 다양한 태스크 적용 가능

감성 분석, 문장 요약, 번역 등 다양한 NLP 태스크에 사용 가능

4. 모델을 새로 저장하지 않아도 됨

LoRA처럼 Plug-and-Play 방식으로 활용 가능

##  6. 활용 사례

1) GPT, BERT 등의 성능 향상

기존 모델을 그대로 사용하면서도 최적의 프롬프트를 학습하여 성능을 향상

2) Zero-shot & Few-shot Learning 강화

적은 데이터로도 더 좋은 결과를 얻을 수 있음

3) RAG (Retrieval-Augmented Generation)에서 활용 가능

검색된 문맥을 최적화된 방식으로 모델에 입력 가능

4) 다양한 NLP 태스크 적용 가능

감성 분석, QA, 문서 요약, 챗봇 등 다양한 활용 가능

## 정리

- Prompt Tuning은 "학습 가능한 프롬프트 벡터(Trainable Prompt Embedding)"를 사용하여 LLM을 최적화하는 기법이다.
- 기존 Full Fine-Tuning보다 훨씬 적은 메모리로도 효과적인 성능을 낼 수 있다.
- Prompt Engineering보다 유연하고, 자동으로 최적화되는 방식이다.
- LoRA, P-Tuning, Prefix-Tuning과 함께 Fine-Tuning을 대체할 수 있는 효율적인 방법이다.

특히 RAG + Prompt Tuning을 함께 활용하면, 검색된 문서를 최적의 형태로 모델에 입력하여 성능을 극대화할 수 있음
