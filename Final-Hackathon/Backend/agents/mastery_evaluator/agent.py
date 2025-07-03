# agents/mastery_evaluator/agent.py

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from agents.mastery_evaluator.prompts import get_mastery_prompt
from config import GEMINI_API_KEY, GEMINI_MODEL

# Gemini agent for mastery evaluation
def create_mastery_agent():
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0.4,
        google_api_key=GEMINI_API_KEY
    )

# Core evaluation logic (do not change)
async def evaluate_mastery_agent(input_data):
    print("coming data", input_data)
    prompt = get_mastery_prompt(
        quiz_scores=input_data.quiz_scores,
        retry_data=input_data.retry_data,
        time_data=input_data.time_data
    )

    model = create_mastery_agent()
    parser = JsonOutputParser()
    result = await model.ainvoke(prompt)
    return parser.invoke(result.content)

# Wrap the function as a LangChain tool
mastery_tool = Tool(
    name="evaluate_mastery_agent",
    func=lambda inputs: evaluate_mastery_agent(inputs["input_data"]),
    description="Evaluates a learner's mastery level using quiz scores, coding logs, retry patterns, and time spent.",
    coroutine=evaluate_mastery_agent
)

# Initialize the agent with the mastery evaluation tool
tools = [mastery_tool]
llm = create_mastery_agent()
agent = initialize_agent(
    tools=tools,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    llm=llm,
    verbose=True
)