import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model='gpt-4o') # gpt-3.5-turbo , gpt-4o
embedding = OpenAIEmbeddings(model='text-embedding-3-large')
index_name = 'lee-index'
database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)
retriever = database.as_retriever(search_kwargs={'k': 4})

system_prompt = PromptTemplate(
    template="""
    당신은 이성규라는 사람을 광적으로 좋아하는 스토커입니다. 사용자의 질문에 답변해주세요. 
    당신이 기록한 문서가 있습니다. 이를 활용해서 답변을 해주세요.
    만약 문서에서 답변을 알 수 없다면 반드시 알아낼 거라고 더 스토킹을 할 것이란 답변해주세요. 
    만약 문서에서 답변을 알 수 있다면 내가 XX일에 관찰한 결과에 따르면 이라고 시작하면서 답변해주세요. 
    만약 이성규외 다른 사람에 관련된 질문이 들어와도 그 사람은 모르겠고 이성규는... 라며 이성규에 관련된 답변을 해주세요. 
    2-3 문장정도의 짧은 내용의 답변을 원합니다.

    {context}

    Question: {question}"""
    , input_variables=["context","question"]
)

rag_chain = system_prompt | llm

st.set_page_config(page_title="이성규 스토커", page_icon="🥷")

st.title("🥷 이성규 스토커")
st.caption("매일 관찰한다고 하네요!")

if 'message_list' not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])


if user_question := st.chat_input(placeholder="궁금한게 있으면 물어보세요!"):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})

    with st.spinner("🕵️스토커가 자료를 찾아보는 중.."):
        ai_response1 = retriever.invoke(user_question)
        ai_response = rag_chain.invoke( {"context" : ai_response1,"question" : user_question})
        with st.chat_message("ai"):
            st.write(ai_response.content)
            st.session_state.message_list.append({"role": "ai", "content": ai_response.content})