# models/learner.py
from pydantic import BaseModel
from typing import Dict, List
from datetime import datetime
from database.connection import db

class MasteryEvaluationInput(BaseModel):
    learner_id: str
    quiz_scores: Dict[str, float]
    coding_logs: Dict[str, int]
    retry_data: Dict[str, str]
    time_data: Dict[str, float]

def log_agent_response(agent_name: str, user_id: str, input_data: dict, response: dict):
    db.agent_responses.insert_one({
        "agent": agent_name,
        "userId": user_id,
        "input": input_data,
        "response": response,
        "timestamp": datetime.utcnow()
    })
