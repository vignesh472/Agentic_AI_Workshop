def get_adaptive_prompt(concept, level):
    return f"""
You are an expert AI tutor. Generate an adaptive conceptual assessment for the concept: "{concept}".

Context:
- The learner has shown "{level.lower()}" understanding of this concept.
- Your task is to probe their depth of understanding with increasing difficulty.

Instructions:
1. Create 3 conceptual questions (levels 1 to 3: easy to deep thinking).
2. For each question, provide:
   - level (1 to 3)
   - question
   - ideal_answer
   - reasoning_prompt (e.g., “Why does this work?”, “What would happen if…”)

Output format (strictly JSON):
{{
  "questions": [
    {{
      "level": 1,
      "question": "What is recursion?",
      "ideal_answer": "A function that calls itself to solve a smaller subproblem.",
      "reasoning_prompt": "Why is a base case necessary in recursion?"
    }},
    ...
  ]
}}
"""

def get_grading_prompt(concept, question, user_answer, ideal_answer):
    return f"""
You are an expert AI tutor grading a student's conceptual understanding.

Concept: "{concept}"
Question: "{question}"
Ideal Answer: "{ideal_answer}"
Student's Answer: "{user_answer}"

Instructions:
1. Evaluate the student's answer using these rubrics (scale 1-5):
   - correctness
   - completeness
   - reasoning clarity

2. Identify any conceptual misunderstandings or reasoning flaws.

3. Suggest clear feedback that helps the learner improve.

Output format (strictly JSON):
{{
  "score": {{
    "correctness": 4,
    "completeness": 3,
    "reasoning": 2
  }},
  "feedback": "The student confuses recursion with iteration. They overlook the base case's role in termination."
}}
"""
