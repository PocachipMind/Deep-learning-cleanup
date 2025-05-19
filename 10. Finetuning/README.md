# 한 권으로 끝내는 실전 LLM 파인튜닝

![image](https://github.com/user-attachments/assets/41f776a1-c2bd-4a2c-a7c5-0b79653582ee)


해당 책 깃https://github.com/wikibook/llm-finetuning



## 

런팟을 통한 클라우드 환경 사용.

report_to 인자를 통해 wandb ( 더블유 앤 비 디비 ) 보는 방법 

### 1. 학습 데이터 : PDF에서 로드해서 청크로 자른 후 학습 데이터로 변환

FSDP, QLoRA

해당 부분은 언슈퍼바이즈드 텍스트 생성 스타일 즉 스타일 모방 훈련 ( prompt -> response 형태가 아님 )

* PDF 청크들과 유사한 말투, 문체, 주제를 가진 텍스트 생성.

사전 학습되어있는 인스트럭션 모델에 튜닝을 해서인지, 인스트럭션 형식으로도 동작이 잘되었음.

### 2. RAG 기반 LLM 최적화 학습

RAFT에서 출처 인용

- 자른 청크를 텍스트로 넣으며 관련된 질문과 답변을 달라하고
- 학습시 청크 잘못된것 ( 네거티브 샘플 ) 과 제대로된 샘플을 넣고 질문을 넣고 학습
- 인스트럭션 튜닝.

### 3. vLLM 
핵심 : 페이지 어텐션

런팟의 vLLM Docker를 활용해 API를 직접 배포하고 구축

런팟의 vLLM Docker를 활용하면 클라우드 환경의 고성능 GPU 리소스를 활용해 vLLM 기반 API를 쉽게 배포할 수 있습니다.

런팟의 Serverless vLLM활용 API로 활용

### 4. vLLM 활용 Multi-LoRA

