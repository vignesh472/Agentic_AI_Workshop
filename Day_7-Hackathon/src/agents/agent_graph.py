from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda

from agents.performance_agent import analyze_performance
from agents.expectation_agent import retrieve_expectations
from agents.gap_agent import explain_gap

# Define state keys
state = {
    "results": str,
    "role": str,
    "level": str,
    "insights": str,
    "expectations": str,
    "gap": str,
}

# Nodes
def performance_node(state):
    insights = analyze_performance(state["results"])
    return {**state, "insights": insights}

def expectation_node(state):
    expectations = retrieve_expectations(state["role"], state["level"])
    return {**state, "expectations": expectations}

def gap_node(state):
    gap = explain_gap(state["insights"], state["expectations"])
    return {**state, "gap": gap}

# Build graph
def build_agent_graph():
    graph = StateGraph(state)

    graph.add_node("PerformanceAnalysis", RunnableLambda(performance_node))
    graph.add_node("RetrieveExpectations", RunnableLambda(expectation_node))
    graph.add_node("ExplainGap", RunnableLambda(gap_node))

    graph.set_entry_point("PerformanceAnalysis")
    graph.add_edge("PerformanceAnalysis", "RetrieveExpectations")
    graph.add_edge("RetrieveExpectations", "ExplainGap")
    graph.set_finish_point("ExplainGap")

    return graph.compile()
