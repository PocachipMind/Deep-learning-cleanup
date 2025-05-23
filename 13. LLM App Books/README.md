# 랭체인과 RAG로 배우는 실전 LLM 애플리케이션 개발
### 멀티모달/GraphRAG/ReAct 에이전트/sLLM 완벽 실무 가이드


![image](https://github.com/user-attachments/assets/0b93c345-4a73-493a-b45e-0d0d3d2abdaf)


https://github.com/wikibook/langchain-rag

https://wikibook.co.kr/langchain-rag/

## 주요 학습 내용
- 랭체인
    - 병렬처리 체인
    - 프롬프트 사용 방법

- Advanced Rag
    - 리랭커 : BM25, TF-IDF
    - HyDE(Hypothetical Document Embeddings)
    - 쿼리 확장 기술 : 통계적, 신경망
    - 멀티 쿼리 기술

- 멀티 모달 RAG
    - 멀티모달 인코더
        - CLIP : 대규모 검색 및 매칭에 적합
        - ViLT : 설명 생성에서 뛰어난 성능
    - 멀티모달 디코더
        - GPT
        - T5
    - RAG
        - DPR
        - FAISS
    - 융합 기술
        - Late Fusion 후기 융합 : 모달리티 특성 보존
        - Cross-Attention : 모달리티 간 상호작용 강화
    - 파인튜닝

- MultiVectorRetriever 활용 멀티모달 RAG : 이미지를 의미하는 텍스트 형성 후 VectorDB에 저장 하여 RAG 실행

- GraphRAG
    - 자연어 쿼리를 통한 그래프 데이터 조회 및 조작
    - LLM 기반 지식 그래프 구축 및 RAG ( Neo4j AuraDB )

- pdf로부터 텍스트 추출하고 그걸 기반으로 그래프 DB 생성, 그래프 리트리버 및 백터 리트리버 실행

- ReAct 에이전트
    - 해당 부분 많이 아쉬움. 추상화되어있는 프레임워크사용이아닌 함수를 연결하여 구성.
    - 해당 부분은 인프런의 랭그래프 강의를 통해 보충.

- sLLM
    - 특정 pdf를 기반으로 파인튜닝 학습 데이터 구성
    - FFT ( Full Fine-Tuning )
        - 병렬 처리 학습 : DP, DDP, FSDP
        - DeepSpeed
    - PEFT ( Parameter-Efficient Fine-Tuning )
        - QLoRA
        - DoRA
    - RAG 기반 LLM 최적화 학습
        - RAFT - 핵심 : 출처 표기 , CoT
    - LLM 서빙
        - Streamlit
        - vLLM

### 6 강 부분 :

#### 1. 학습 데이터 : PDF에서 로드해서 청크로 자른 후 학습 데이터로 변환

FSDP, QLoRA

해당 부분은 언슈퍼바이즈드 텍스트 생성 스타일 즉 스타일 모방 훈련 ( prompt -> response 형태가 아님 )

* PDF 청크들과 유사한 말투, 문체, 주제를 가진 텍스트 생성.

사전 학습되어있는 인스트럭션 모델에 튜닝을 해서인지, 인스트럭션 형식으로도 동작이 잘되었음.

#### 2. RAG 기반 LLM 최적화 학습
RAFT에서 출처 인용

- 자른 청크를 텍스트로 넣으며 관련된 질문과 답변을 달라하고
- 학습시 청크 잘못된것 ( 네거티브 샘플 ) 과 제대로된 샘플을 넣고 질문을 넣고 학습
- 인스트럭션 튜닝.
