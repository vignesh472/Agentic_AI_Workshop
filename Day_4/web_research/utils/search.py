import os
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()
client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def search_web(query):
    results = client.search(query=query, search_depth="advanced", max_results=3)
    return [{"title": r["title"], "content": r["content"]} for r in results.get("results", [])]
