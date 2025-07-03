# agents/prerequisite_detector/agent.py

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from config import GEMINI_API_KEY, GEMINI_MODEL
from agents.prerequisite_detector.prompts import get_dynamic_prerequisite_prompt

# Agent creator
def create_prerequisite_detector_agent():
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0.4,
        google_api_key=GEMINI_API_KEY
    )

# Core logic to detect prerequisite gaps
async def detect_prerequisite_gaps(weak_concepts: list):
    prompt = get_dynamic_prerequisite_prompt(weak_concepts)
    model = create_prerequisite_detector_agent()
    parser = JsonOutputParser()

    result = await model.ainvoke(prompt)
    return parser.invoke(result.content)

# Wrap as LangChain tool
prerequisite_tool = Tool(
    name="detect_prerequisite_gaps",
    func=lambda inputs: detect_prerequisite_gaps(inputs["weak_concepts"]),
    description="Detects prerequisite knowledge gaps for a given list of weak concepts.",
    coroutine=detect_prerequisite_gaps
)

# Initialize the agent with the tool
tools = [prerequisite_tool]
llm = create_prerequisite_detector_agent()
agent = initialize_agent(
    tools=tools,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    llm=llm,
    verbose=True
)