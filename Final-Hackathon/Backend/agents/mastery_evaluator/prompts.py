# agents/mastery_evaluator/prompts.py
def get_mastery_prompt(quiz_scores, coding_logs, retry_data, time_data):
    return f"""
You are an expert AI tutor. Analyze the learner's performance to determine mastery of each concept.

Data:
- Quiz Scores: {quiz_scores}
- Code Attempt Logs: {coding_logs}
- Retry Patterns: {retry_data}
- Time Spent per Concept: {time_data}

Instructions:
1. For each concept, classify the mastery level as:
   - "Strong"
   - "Moderate"
   - "Weak"

2. Give brief reasoning for each classification.

Return output in JSON format like:
{{
  "concepts": [
    {{
      "name": "Loops",
      "mastery": "Strong",
      "reason": "High quiz score and low retry count"
    }},
    ...
  ]
}}
"""
