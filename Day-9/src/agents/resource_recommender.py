import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

def recommend_resources(gap_explanation):
    template = """Given this skill gap: {gap},
    recommend 5 high-quality resources (videos, problems, tutorials) to improve."""

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    prompt = PromptTemplate.from_template(template)
    formatted_prompt = prompt.format(gap=gap_explanation)
    
    return llm.invoke(formatted_prompt)
