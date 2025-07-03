from langgraph.graph import StateGraph, END
from typing import Dict, Any
from agents.mastery_evaluator.agent import evaluate_mastery_agent
from agents.prerequisite_detector.agent import detect_prerequisite_gaps
from agents.booster_recommender.agent import generate_booster_recommendations


# Node: Mastery Evaluation
async def mastery_node(state: Dict[str, Any]) -> Dict[str, Any]:
    input_data = state["input_data"]
    result = await evaluate_mastery_agent(input_data)
    state["mastery_result"] = result

    # Check for Weak or Moderate topics
    weak_or_moderate = [
        c for c in result.get("concepts", []) if c["mastery"] in ["Weak", "Moderate"]
    ]
    if not weak_or_moderate:
        state["message"] = "No weak or moderate topics found. You're all set!"
        state["end_flow"] = True  # Used to exit the graph early
    return state


# Node: Prerequisite Gap Detection
async def gap_node(state: Dict[str, Any]) -> Dict[str, Any]:
    mastery_result = state["mastery_result"]
    weak_concepts = [
        c["name"]
        for c in mastery_result["concepts"]
        if c["mastery"] in ["Weak", "Moderate"]
    ]
    result = await detect_prerequisite_gaps(weak_concepts)
    state["gaps_result"] = result
    return state


# Node: Booster Recommendation
async def booster_node(state: Dict[str, Any]) -> Dict[str, Any]:
    gaps = state["gaps_result"]
    # print("Gaps detected:", gaps)
    result = await generate_booster_recommendations(gaps,category={}, assessmentResult={} ,preference="text")
    state["booster_result"] = result
    return state


# Graph Builder
def build_mastery_graph():
    graph = StateGraph(dict)

    # Add nodes
    graph.add_node("mastery", mastery_node)
    graph.add_node("gaps", gap_node)
    graph.add_node("booster", booster_node)

    # Conditional transition after mastery
    def route_after_mastery(state: Dict[str, Any]) -> str:
        return END if state.get("end_flow") else "gaps"

    graph.add_conditional_edges("mastery", route_after_mastery)

    # Normal transitions
    graph.add_edge("gaps", "booster")
    graph.add_edge("booster", END)

    # Set entry point
    graph.set_entry_point("mastery")

    return graph.compile()
