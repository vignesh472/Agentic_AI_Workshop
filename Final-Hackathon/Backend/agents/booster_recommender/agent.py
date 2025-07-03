# agents/booster_recommender/agent.py

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from config import GEMINI_API_KEY, GEMINI_MODEL
from agents.booster_recommender.prompts import get_booster_prompt

# Create Gemini-powered LLM agent
def create_booster_agent():
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0.5,
        google_api_key=GEMINI_API_KEY
    )

# Function that generates personalized booster recommendations
# Function to generate booster content recommendations
async def generate_booster_recommendations(concepts: list, preference: str, category: list, assessmentResult= list):
    # print("Generating boosters for:")
    # print("Concepts:", concepts)
    # print("Preference:", preference)
    # print("Categories:", category)

    # Step 1: Generate prompt for booster agent
    prompt = get_booster_prompt(concepts, preference, category, assessmentResult)

    # Step 2: Create the booster agent (LLM wrapper)
    model = create_booster_agent()

    # Step 3: Use output parser to get clean JSON response
    parser = JsonOutputParser()

    # Step 4: Call the model and parse the result
    result = await model.ainvoke(prompt)
    boosters = parser.invoke(result.content)

    return boosters

# Wrap as LangChain tool
booster_tool = Tool(
    name="generate_booster_recommendations",
    func=lambda inputs: generate_booster_recommendations(inputs["concepts"], inputs["preference"]),
    description="Generates tailored learning boosters based on list of 'concepts' and a learner's 'preference'.",
    coroutine=generate_booster_recommendations
)

# Initialize the CAT agent with the booster tool
tools = [booster_tool]
llm = create_booster_agent()
agent = initialize_agent(
    tools=tools,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    llm=llm,
    verbose=True
)