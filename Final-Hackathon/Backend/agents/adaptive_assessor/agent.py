from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
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
async def generate_adaptive_questions(concept: str, level: str,category:str):
    prompt = get_adaptive_prompt(concept, level,category)
    model = create_adaptive_agent()
    parser = JsonOutputParser()
    result = await model.ainvoke(prompt)
    return parser.invoke(result.content)

# 2. Grade the user's answer based on ideal answer
async def grade_answer(concept: str, question: str, user_answer: str, ideal_answer: str,question_data: list,user_selected_option: list):
    prompt = get_grading_prompt(concept, question, user_answer, ideal_answer,question_data,user_selected_option)
    model = create_adaptive_agent()
    parser = JsonOutputParser()
    result = await model.ainvoke(prompt)
    return parser.invoke(result.content)

# Wrapping functions as LangChain tools
adaptive_question_tool = Tool(
    name="generate_adaptive_questions",
    func=lambda inputs: generate_adaptive_questions(inputs["concept"], inputs["level"]),
    description="Use this tool to generate 3-level adaptive conceptual questions. Requires 'concept' and 'level'.",
    coroutine=generate_adaptive_questions
)

grading_tool = Tool(
    name="grade_answer",
    func=lambda inputs: grade_answer(
        inputs["concept"], inputs["question"], inputs["user_answer"], inputs["ideal_answer"]
    ),
    description="Use this tool to grade the learner's answer. Requires 'concept', 'question', 'user_answer', and 'ideal_answer'.",
    coroutine=grade_answer
)

# Initialize the agent with tools
tools = [adaptive_question_tool, grading_tool]
llm = create_adaptive_agent()
agent = initialize_agent(
    tools=tools,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    llm=llm,
    verbose=True
)