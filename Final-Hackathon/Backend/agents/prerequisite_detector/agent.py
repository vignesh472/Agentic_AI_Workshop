# agents/prerequisite_detector/agent.py

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from config import GEMINI_API_KEY, GEMINI_MODEL
from agents.prerequisite_detector.prompts import get_dynamic_prerequisite_prompt

def create_prerequisite_detector_agent():
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0.4,
        google_api_key=GEMINI_API_KEY
    )

async def detect_prerequisite_gaps(weak_concepts: list):
    prompt = get_dynamic_prerequisite_prompt(weak_concepts)
    model = create_prerequisite_detector_agent()
    parser = JsonOutputParser()

    result = await model.ainvoke(prompt)
    return parser.invoke(result.content)
