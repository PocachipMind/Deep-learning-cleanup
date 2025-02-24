# RAG 관련

공식 강의 자료 :

https://github.com/jasonkang14/inflearn-rag-notebook


올려진 자료는 강의에서 추가적으로 자료가 재정리되어있음.

고로 올려진 자료를 그대로 보는 것 보단 나 스스로가 다시 볼 때 유용하게 재정리하여 기재한 Git.


https://python.langchain.com/api_reference/langchain/chains/langchain.chains.retrieval_qa.base.RetrievalQA.html

4.3의 QA가 더이상 안쓰인다고 한다.

# Chat App 관련

https://github.com/jasonkang14/inflearn-streamlit-lecture

# 메모

업스테이지 임베딩이 한국어는 더 잘되는 경향이 있다.

-----------------
텍스트 스플리터는 다양한데, 일반적으로 문서는 Recursive Character Text Splitter나 Character Text Splitter를 많이씀

차이점 : Character Text Splitter는 구분자를 하나밖에 못 넣음 그래서 두줄 띄어있는거 단위로 잘라버리는데 Recursive 경우는 리스트로 구분자를 줄 수 있음. 그래서 Recursive 로 하시는게 더 성능이 좋다.

-----------------

질문할 때 Please 즉 존대말로 하면 성능이 좋다는 논문이있음. 존대말로 질문할것.

---------------

페르소나를 주면 성능이 올라가는 경우가 많다. 

[Identity]
- 당신은 최고의 ...전문가입니다...
- [Context]를 참고해서 사용자의 질문에 답변해주세요.

[Context]
{}

question: {}

---------------------

유사도 : 
- 유클리디안 : 거리
- dot product : 클수록 유사하다
- cosine : 벡터사이 각도 

---------------------

데이터 전처리하는 것도 중요하지만 유사도 검색을 할 때 어떤 문장을 활용할지도 굉장히 중요하다.

그래서 회사에서 하신다면 RAG로 활용되는 문서를 보시고 빈번하게 사용될만한 단어를 찾아서 정리를 해주셔야 됩니다.
