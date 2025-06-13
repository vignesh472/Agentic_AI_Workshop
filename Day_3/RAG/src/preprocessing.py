import fitz
import numpy as np
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_and_chunk_pdfs(folder_path="data"):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            path = os.path.join(folder_path, filename)
            doc = fitz.open(path)
            for page in doc:
                text = page.get_text()
                # Add page metadata at the end
                text += f"\nSOURCE: {filename} PAGE: {page.number+1}"
                documents.append((text, filename, page.number+1))
    return documents

def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = []
    for text, filename, page_num in documents:
        chunks += [
            (chunk, filename, page_num) 
            for chunk in text_splitter.split_text(text)
        ]
    return chunks