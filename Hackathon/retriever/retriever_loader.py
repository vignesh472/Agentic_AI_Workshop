from langchain_community.vectorstores import FAISS
from retriever.embedder import get_embedder

def load_retriever(index_path="guides_index"):
    db = FAISS.load_local(
        index_path,
        embeddings=get_embedder(),
        allow_dangerous_deserialization=True  # ⚠️ Only if you trust the source!
    )
    return db.as_retriever()
