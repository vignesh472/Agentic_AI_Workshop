def get_mastery_prompt(quiz_scores, retry_data, time_data):
    return f"""
You are an expert AI tutor. Analyze the learner's performance data to determine mastery of each concept.

Performance Data:
- Quiz Scores: {quiz_scores}
- Retry Patterns: {retry_data}
- Time Spent per Concept: {time_data}

Instructions:
For each concept, classify the mastery level as:
- "Strong"
- "Moderate"
- "Weak"

Use the following logic:
- High quiz score (≥80), low retries (≤1), and low time spent = Strong
- Medium quiz score (50–79), moderate retries (2–3), or average time = Moderate
- Low quiz score (<50), high retries (>3), or excessive/very low time = Weak

Return your analysis in this JSON format:
{{
  "concepts": [
    {{
      "name": "ConceptName",
      "mastery": "Strong" | "Moderate" | "Weak",
      "reason": "Brief explanation based on quiz score, retry count, and time spent"
    }},
    ...
  ]
}}
"""
