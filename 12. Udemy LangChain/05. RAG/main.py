import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough


load_dotenv()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == "__main__":
    print(" Retrieving...")

    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI()

    query = "what is Pinecone in machine learning?"
    chain = PromptTemplate.from_template(template=query) | llm
    # result = chain.invoke(input={})
    # print(result.content)

    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrival_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )

    result = retrival_chain.invoke(input={"input": query})

    print(result)

    template = """Use the follwing pieces of context to answer the question at the end.
    if you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "thanks for asking!" at the end of the answer.
    
    {context}
    
    Question : {question}
    
    Helpful Answer:"""
    custom_rag_prompt = PromptTemplate.from_template(template)
    
    rag_chain = (
        {"context" : vectorstore.as_retriever() | format_docs, "question" : RunnablePassthrough()}
        | custom_rag_prompt
        | llm
    )
    
    res = rag_chain.invoke(query)
    