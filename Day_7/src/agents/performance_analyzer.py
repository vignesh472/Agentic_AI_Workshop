import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# Load environment variables from .env
load_dotenv()

def analyze_performance(results):
    template = """You are a tutor. Analyze this learner performance data: {results}.
    Identify the top-3 weak areas with reasons."""

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",  
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    prompt = PromptTemplate.from_template(template)
    formatted_prompt = prompt.format(results=results)
    
    return llm.invoke(formatted_prompt)
