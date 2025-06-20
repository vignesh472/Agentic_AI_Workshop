# agents/mastery_evaluator/agent.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from agents.mastery_evaluator.prompts import get_mastery_prompt
from config import GEMINI_API_KEY, GEMINI_MODEL

def create_mastery_agent():
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0.4,
        google_api_key=GEMINI_API_KEY
    )

async def evaluate_mastery_agent(input_data):
    prompt = get_mastery_prompt(
        quiz_scores=input_data.quiz_scores,
        coding_logs=input_data.coding_logs,
        retry_data=input_data.retry_data,
        time_data=input_data.time_data
    )

    model = create_mastery_agent()
    parser = JsonOutputParser()
    result = await model.ainvoke(prompt)
    return parser.invoke(result.content)
