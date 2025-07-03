from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from config import GEMINI_API_KEY, GEMINI_MODEL
from agents.prerequisite_retriever.vectorstore import get_vectorstore
from agents.prerequisite_retriever.prompts import get_rag_summary_prompt

def create_retriever_agent():
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0.4,
        google_api_key=GEMINI_API_KEY
    )

async def retrieve_prerequisites(concepts: list):
    vs = get_vectorstore()
    model = create_retriever_agent()
    parser = JsonOutputParser()
    results = []

    for concept in concepts:
        docs = vs.similarity_search(concept, k=3)
        prompt = get_rag_summary_prompt(concept, docs)
        result = await model.ainvoke(prompt)
        parsed = parser.invoke(str(result.content))
        results.append(parsed)

    return results
