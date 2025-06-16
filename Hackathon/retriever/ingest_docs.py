
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from retriever.embedder import get_embedder
import os

def ingest_docs(pdf_path="guides/uber_dsa.pdf", save_path="guides_index"):
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    
    db = FAISS.from_documents(chunks, embedding=get_embedder())
    db.save_local(save_path)

if __name__ == "__main__":
    ingest_docs()
