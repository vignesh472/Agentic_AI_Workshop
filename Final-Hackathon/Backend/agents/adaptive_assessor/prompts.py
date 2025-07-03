# def get_adaptive_prompt(concept, level,category):
#     print(f"Generating adaptive questions for concept: {concept}, level: {level}, category: {category}")
#     return f"""
# You are an expert AI tutor. Generate an adaptive conceptual assessment for the concept: "{concept}" with the category:"{category}".

# Context:
# - The learner has shown "{level.lower()}" understanding of this concept.
# - Your task is to probe their depth of understanding with increasing difficulty.

# Instructions:
# 1. Create 3 conceptual questions (levels 1 to 3: easy to deep thinking).
# 2. For each question, provide:
#    - level (1 to 3)
#    - question
#    - ideal_answer
#    - reasoning_prompt (e.g., “Why does this work?”, “What would happen if…”)

# Output format (strictly JSON):
# {{
#   "questions": [
#     {{
#       "level": 1,
#       "question": "What is recursion?",
#       "ideal_answer": "A function that calls itself to solve a smaller subproblem.",
#       "reasoning_prompt": "Why is a base case necessary in recursion?"
#     }},
#     ...
#   ]
# }}
# """


def get_adaptive_prompt(concept, level, category):
    print(f"Generating adaptive questions for concept: {concept}, level: {level}, category: {category}")
    return f"""
You are an expert AI tutor. Generate an **adaptive multiple-choice conceptual assessment** for the concept: "{concept}" under the category: "{category}".

Context:
- The learner has shown "{level.lower()}" understanding of this concept.
- Your task is to probe their depth of understanding with increasing difficulty.

Instructions:
1. Create 3 multiple-choice conceptual questions (levels 1 to 3: easy to deep thinking).
2. For each question, include:
   - level (1 to 3)
   - question
   - options (list of 4 answer choices labeled 'A', 'B', 'C', 'D')
   - correct_option (e.g., 'A', 'B', etc.)
   - explanation (why the correct option is right)
   - reasoning_prompt (e.g., “Why does this work?”, “What would happen if…”)

Output format (strictly in JSON):
{{
  "questions": [
    {{
      "level": 1,
      "question": "Which of the following best describes recursion?",
      "options": {{
        "A": "A loop that runs until a condition is met.",
        "B": "A function calling itself to solve a smaller problem.",
        "C": "Using global variables for solving problems.",
        "D": "A type of iterative statement in programming."
      }},
      "correct_option": "B",
      "explanation": "Recursion involves a function calling itself with smaller inputs.",
      "reasoning_prompt": "Why is a base case necessary in recursion?"
    }},
    ...
  ]
}}
"""

def get_grading_prompt(concept, question, user_answer, ideal_answer, question_data, user_selected_option):
    print(f"Grading answer for concept: {concept}, question: {question}")
    print(f"User's answer: {user_answer}")
    print(f"Ideal answer: {ideal_answer}")  
    print(f"Question data: {question_data}")
    print(f"User selected option: {user_selected_option}")
    
    options = question_data.get("options", {})
    option_texts = "\n".join([f"{key}: {value}" for key, value in options.items()])
    
    correct_option = question_data.get("correct_option", "")
    correct_answer_text = options.get(correct_option, "N/A")
    student_answer_text = options.get(user_answer, user_selected_option.get("value", "N/A"))
    explanation = question_data.get("explanation", "")

    return f"""
You are an expert AI tutor evaluating a student's understanding of a computer science concept.

Concept: "{concept}"
Question: "{question}"

Available Options:
{option_texts}

Correct Option: "{correct_option}" — "{correct_answer_text}"
Student's Selected Option: "{user_answer}" — "{student_answer_text}"

Explanation for Correct Answer: "{explanation}"

Instructions:
1. Compare the student's answer (full text) with the correct answer.
2. Evaluate on a scale of 1 to 5 for each of the following:
   - correctness: Is the answer factually and logically accurate?
   - completeness: Does the answer cover all key aspects expected?
   - reasoning: Is the explanation or reasoning coherent and well-structured?

3. Decide whether the student's answer is overall **correct or incorrect** (based on correctness >= 4 and completeness >= 4).

4. If incorrect, identify the main misconceptions or missing parts.

5. Suggest clear, specific feedback to help the student improve.

Output (strictly in valid JSON format):

{{
  "score": {{
    "correctness": <int between 1 and 5>,
    "completeness": <int between 1 and 5>,
    "reasoning": <int between 1 and 5>
  }},
  "is_correct": <true or false>,
  "feedback": "<brief feedback text>"
}}
"""
