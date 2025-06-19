from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import os

def explain_gap(insights, expectations):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=os.getenv("GOOGLE_API_KEY"))

    prompt = PromptTemplate.from_template(
        """Given learner's weak areas: {insights}
        and industry expectations: {expectations}, explain the gap and its impact."""
    )
    return llm.invoke(prompt.format(insights=insights, expectations=expectations))
