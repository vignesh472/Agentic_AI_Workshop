import os
import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from utils.common import load_embeddings

PDF_PATH = "data/industry_roles_dataset.pdf"

def build_index():
    docs = []

    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"{PDF_PATH} not found")

    with fitz.open(PDF_PATH) as doc:
        for page in doc:
            text = page.get_text()
            entries = text.strip().split("\n\n")  # rough grouping per role
            for entry in entries:
                if "Role:" in entry and "Company:" in entry and "Level:" in entry:
                    docs.append(Document(page_content=entry.strip()))

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
    chunks = splitter.split_documents(docs)

    embeddings = load_embeddings()
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("rag_index.faiss")


def get_retriever():
    db = FAISS.load_local(
        "rag_index.faiss",
        load_embeddings(),
        allow_dangerous_deserialization=True
    )
    return db.as_retriever()
