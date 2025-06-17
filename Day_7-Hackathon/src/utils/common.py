import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

import os

def load_llm():
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))



from langchain_google_genai import GoogleGenerativeAIEmbeddings

def load_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def init_vectorstore(embeddings, path="rag_index.faiss"):
    if os.path.exists(path):
        return FAISS.load_local(path, embeddings)
    else:
        return None
