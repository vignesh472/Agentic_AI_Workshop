import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

def explain_gap(insights, expectations):
    template = """Given learner's weak areas: {insights}
    and industry expectations: {expectations}, explain the gap and its impact."""
    
    llm = ChatGoogleGenerativeAI(
        model="models/chat-bison-001",  # Or "gemini-pro" if you're using that
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    prompt = PromptTemplate.from_template(template)
    formatted_prompt = prompt.format(insights=insights, expectations=expectations)
    
    return llm.invoke(formatted_prompt)
