from langchain_google_genai import GoogleGenerativeAIEmbeddings

import os
from dotenv import load_dotenv

load_dotenv()

def get_embedder():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
