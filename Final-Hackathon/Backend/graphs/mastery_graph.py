from langgraph.graph import StateGraph, END
from typing import Dict, Any
from agents.mastery_evaluator.agent import evaluate_mastery_agent
from agents.prerequisite_detector.agent import detect_prerequisite_gaps
from agents.booster_recommender.agent import generate_booster_recommendations

# Node: Mastery Evaluation
def mastery_node(state):
    input_data = state["input_data"]
    result = yield evaluate_mastery_agent(input_data)
    state["mastery_result"] = result
    return state

# Node: Prerequisite Gap Detection
def gap_node(state):
    mastery_result = state["mastery_result"]
    weak_concepts = [c["name"] for c in mastery_result["concepts"] if c["mastery"] == "Weak"]
    result = yield detect_prerequisite_gaps(weak_concepts)
    state["gaps_result"] = result
    return state

# Node: Booster Recommendation
def booster_node(state):
    gaps = state["gaps_result"]
    # For demo, use 'text' as default preference
    result = yield generate_booster_recommendations(gaps, "text")
    state["booster_result"] = result
    return state

# Build the LangGraph workflow
def build_mastery_graph():
    graph = StateGraph(dict)  # Use dict as the state schema
    graph.add_node("mastery", mastery_node)
    graph.add_node("gaps", gap_node)
    graph.add_node("booster", booster_node)
    graph.add_edge("mastery", "gaps")
    graph.add_edge("gaps", "booster")
    graph.add_edge("booster", END)
    graph.set_entry_point("mastery")
    return graph.compile()
