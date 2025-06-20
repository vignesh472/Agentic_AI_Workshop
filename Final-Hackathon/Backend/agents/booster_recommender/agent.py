# agents/booster_recommender/agent.py

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from config import GEMINI_API_KEY, GEMINI_MODEL
from agents.booster_recommender.prompts import get_booster_prompt

def create_booster_agent():
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0.5,
        google_api_key=GEMINI_API_KEY
    )

async def generate_booster_recommendations(concepts: list, preference: str):
    prompt = get_booster_prompt(concepts, preference)
    model = create_booster_agent()
    parser = JsonOutputParser()

    result = await model.ainvoke(prompt)
    return parser.invoke(result.content)
