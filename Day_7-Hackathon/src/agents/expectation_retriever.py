from rag.rag_index import get_retriever
from langchain.chains import RetrievalQA
from utils.common import load_llm
import re

def retrieve_expectations(role, level):
    retriever = get_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=load_llm(), retriever=retriever)

    query = f"What are the skill expectations for a {role} at {level}?"
    result = qa_chain.run(query)

    return result

def extract_skills_from_text(text):
    """
    Extracts the skills from retrieved expectation text.
    Looks for line starting with 'Skills:' and splits by commas.
    """
    match = re.search(r"Skills:\s*(.+)", text, re.IGNORECASE)
    if match:
        return [s.strip() for s in match.group(1).split(",")]
    return []
