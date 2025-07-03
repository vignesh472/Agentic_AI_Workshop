from pathlib import Path
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import GEMINI_API_KEY
from PyPDF2 import PdfReader
from config import GEMINI_MODEL  

embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GEMINI_API_KEY
)

# This module handles loading PDF documents, splitting them into chunks, and creating a vector store for retrieval.
def load_corpus_from_pdfs(folder="data/"):
    docs = []
    for file in Path(folder).glob("*.pdf"):
        reader = PdfReader(file)
        content = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        if content.strip():
            concept = file.stem
            docs.append(Document(page_content=content, metadata={"concept": concept}))
    return docs


def get_vectorstore():
    docs = load_corpus_from_pdfs()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    return FAISS.from_documents(chunks, embedding)
