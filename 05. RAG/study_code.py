from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv

from fewshot import answer_examples


# 공통 변수 설정 

load_dotenv()

dictionary = ["리그오브레전드를 표현하는 단어-> 롤","주어가 없는 문장 -> 이성규를 주어인 문장"]
llm = ChatOpenAI(model='gpt-3.5-turbo') # gpt-4o

store = {}



# get_dictionary_chain
prompt = ChatPromptTemplate.from_template(f"""
    사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
    만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
    그런 경우에는 질문만 리턴해주세요
    사전: {dictionary}
    
    질문: {{question}}
""")

dictionary_chain = prompt | llm | StrOutputParser()




# get_history_retriever
embedding = OpenAIEmbeddings(model='text-embedding-3-large')
index_name = 'lee-index'
database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)
retriever = database.as_retriever(search_kwargs={'k': 4})

# https://python.langchain.com/v0.2/docs/how_to/qa_chat_history_how_to/#prompt
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# get_rag_chain
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{answer}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=answer_examples,
)
system_prompt = (
    "당신은 이성규라는 사람을 광적으로 좋아하는 스토커입니다. 사용자의 질문에 답변해주세요"
    "아래에 제공된 문서를 활용해서 답변해주시고"
    "만약 제공된 문서에서 답변을 전혀 알 수 없다면 반드시 알아낼 거라고 더 스토킹을 할 것이란 답변해주세요"
    "만약 이성규외 다른 사람에 관련된 질문이 들어와도 그 사람은 모르겠고 이성규는... 라며 이성규에 관련된 답변을 해주세요"
    "답변을 제공할 때는 내가 XX일에 관찰한 결과에 따르면 이라고 시작하면서 답변해주시고"
    "2-3 문장정도의 짧은 내용의 답변을 원합니다"
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        few_shot_prompt,
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
).pick('answer')




user_message = "롤을 좋아해?"

tax_chain = {"input": dictionary_chain} | rag_chain
ai_message = tax_chain.invoke(
    {
        "question": user_message
    },
    config={
        "configurable": {"session_id": "abc123"}
    },
)

print(ai_message)
