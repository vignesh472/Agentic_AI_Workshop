from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import os

def analyze_performance(results):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=os.getenv("GOOGLE_API_KEY"))

    prompt = PromptTemplate.from_template(
        """You are a tutor. Analyze this learner performance data: {results}.
        Identify the top-3 weak areas with reasons."""
    )
    return llm.invoke(prompt.format(results=results))
