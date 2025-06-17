from rag.rag_index import get_retriever
from langchain.chains import RetrievalQA
from utils.common import load_llm

def retrieve_expectations(role, level):
    retriever = get_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=load_llm(), retriever=retriever)
    return qa_chain.run(f"What are the skill expectations for a {role} at {level}?")
