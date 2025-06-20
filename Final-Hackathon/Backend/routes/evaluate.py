# routes/evaluate.py
from fastapi import APIRouter
from database.models.learner import MasteryEvaluationInput, log_agent_response
from agents.mastery_evaluator.agent import evaluate_mastery_agent
from agents.prerequisite_detector.agent import detect_prerequisite_gaps
from agents.adaptive_assessor.agent import generate_adaptive_questions, grade_answer
from agents.booster_recommender.agent import generate_booster_recommendations
from agents.prerequisite_retriever.agent import retrieve_prerequisites
from fastapi import Body
from graphs.mastery_graph import build_mastery_graph

router = APIRouter()

@router.post("/mastery")
async def evaluate_mastery(input_data: MasteryEvaluationInput):
    result = await evaluate_mastery_agent(input_data)
    log_agent_response(
        agent_name="mastery_evaluator",
        user_id="test_user_001",
        input_data=input_data.dict(),
        response=result
    )
    return {"status": "success", "evaluation": result}


@router.post("/prerequisite-gaps")
async def prerequisite_gap_detector(weak_concepts: list = Body(...)):
    result = await detect_prerequisite_gaps(weak_concepts)
    log_agent_response(
        agent_name="prerequisite_detector",
        user_id="test_user_001",
        input_data={"weak_concepts": weak_concepts},
        response=result
    )
    return {"status": "success", "gaps": result}


# @router.post("/adaptive-assess")
# async def adaptive_question_generator(
#     body: dict = Body(...)
# ):
#     concept = body.get("concept")
#     level = body.get("level", "moderate")  # default fallback
#     user_id = "test_user_001"  # Static user ID
#     if concept is None:
#         return {"status": "error", "message": "'concept' is required."}
#     result = await generate_adaptive_questions(concept, level)
#     log_agent_response(
#         agent_name="adaptive_assessor",
#         user_id=user_id,
#         input_data={"concept": concept, "level": level},
#         response=result
#     )
#     return {"status": "success", "adaptive_questions": result}




@router.post("/adaptive/questions")
async def get_questions(
    concept: str = Body(...), 
    level: str = Body(...)
):
    print("come in ✅")  # This should show in the terminal
    return await generate_adaptive_questions(concept, level)

@router.post("/adaptive/grade")
async def get_grade(
    concept: str = Body(...),
    question: str = Body(...),
    user_answer: str = Body(...),
    ideal_answer: str = Body(...)
):
    return await grade_answer(concept, question, user_answer, ideal_answer)

@router.post("/booster-recommend")
async def booster_recommendation_endpoint(
    body: dict = Body(...)
):
    concepts = body.get("concepts", [])
    preference = body.get("preference", "text")

    if not concepts:
        return {"status": "error", "message": "No concepts provided."}

    result = await generate_booster_recommendations(concepts, preference)
    log_agent_response(
        agent_name="booster_recommender",
        user_id="test_user_001",
        input_data={"concepts": concepts, "preference": preference},
        response=result
    )
    return {"status": "success", "boosters": result}


@router.post("/retrieve-prerequisite")
async def retrieve_prerequisite_material(body: dict = Body(...)):
    concepts = body.get("concepts", [])
    if not concepts:
        return {"status": "error", "message": "Concept list is empty."}

    result = await retrieve_prerequisites(concepts)
    log_agent_response(
        agent_name="prerequisite_retriever",
        user_id="test_user_001",
        input_data={"concepts": concepts},
        response={"retrieved": result}
    )
    return {"status": "success", "retrieved": result}

@router.post("/agentic/mastery-graph")
async def agentic_mastery_graph(input_data: MasteryEvaluationInput):
    graph = build_mastery_graph()
    # The state must be a dict with 'input_data' key
    state = {"input_data": input_data}
    # Run the graph using the correct async method
    final_state = await graph.ainvoke(state)
    return {"status": "success", "results": final_state}
