from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from config import GEMINI_API_KEY, GEMINI_MODEL
from agents.adaptive_assessor.prompts import get_adaptive_prompt, get_grading_prompt

def create_adaptive_agent():
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0.6,
        google_api_key=GEMINI_API_KEY,
        streaming=False
    )

# 1. Generate 3-level adaptive conceptual questions
async def generate_adaptive_questions(concept: str, level: str):
    prompt = get_adaptive_prompt(concept, level)
    model = create_adaptive_agent()
    parser = JsonOutputParser()
    result = await model.ainvoke(prompt)
    return parser.invoke(result.content)

# 2. Grade the user's answer based on ideal answer
async def grade_answer(concept: str, question: str, user_answer: str, ideal_answer: str):
    prompt = get_grading_prompt(concept, question, user_answer, ideal_answer)
    model = create_adaptive_agent()
    parser = JsonOutputParser()
    result = await model.ainvoke(prompt)
    return parser.invoke(result.content)
