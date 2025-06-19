from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda

from agents.performance_analyzer import analyze_performance
from agents.expectation_retriever import retrieve_expectations
from agents.gap_explainer import explain_gap
from agents.resource_recommender import recommend_resources  

# Define the graph state schema (keys used in agent chaining)
state = {
    "results": str,
    "role": str,
    "level": str,
    "insights": str,
    "expectations": str,
    "gap": str,
    "resources": str,  
}

# --- Node Definitions ---

def performance_node(state):
    insights = analyze_performance(state["results"])
    return {**state, "insights": insights}

def expectation_node(state):
    expectations = retrieve_expectations(state["role"], state["level"])
    return {**state, "expectations": expectations}

def gap_node(state):
    gap = explain_gap(state["insights"], state["expectations"])
    return {**state, "gap": gap}

def resource_node(state):
    resources = recommend_resources(state["gap"])
    return {**state, "resources": resources}

# --- Build Agent Graph ---

def build_agent_graph():
    graph = StateGraph(state)

    graph.add_node("PerformanceAnalysis", RunnableLambda(performance_node))
    graph.add_node("RetrieveExpectations", RunnableLambda(expectation_node))
    graph.add_node("ExplainGap", RunnableLambda(gap_node))
    graph.add_node("RecommendResources", RunnableLambda(resource_node))  # ✅ Added

    graph.set_entry_point("PerformanceAnalysis")
    graph.add_edge("PerformanceAnalysis", "RetrieveExpectations")
    graph.add_edge("RetrieveExpectations", "ExplainGap")
    graph.add_edge("ExplainGap", "RecommendResources")  # ✅ Chain extended
    graph.set_finish_point("RecommendResources")        # ✅ Set finish point

    return graph.compile()
