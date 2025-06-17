import os
import json
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from utils.common import load_embeddings
def build_index():
    docs = []
    with open("data/industry_benchmarks/sample_roles.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            text = f"Role: {obj['role']}, Company: {obj['company']}, Level: {obj['level']}, Skills: {', '.join(obj['skills'])}"
            docs.append(Document(page_content=text))

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
    chunks = splitter.split_documents(docs)

    embeddings = load_embeddings()
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("rag_index.faiss")



def get_retriever():
    db = FAISS.load_local(
        "rag_index.faiss",
        load_embeddings(),
        allow_dangerous_deserialization=True  # <- add this line
    )
    return db.as_retriever()
