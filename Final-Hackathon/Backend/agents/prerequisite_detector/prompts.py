# agents/prerequisite_detector/prompts.py

def get_dynamic_prerequisite_prompt(weak_concepts: list):
    return f"""
You are an AI tutor specialized in computer science education.

The learner is struggling with the following concepts:
{weak_concepts}

Your task:
1. For each weak concept, identify the foundational concepts or prerequisites that must be understood first.
2. These prerequisites may be direct or indirect (recursive dependencies).
3. Deduplicate all concepts across the list.
4. Focus only on computing, coding, and algorithmic reasoning.

📌 Return the result as a JSON array like:
[
  "Variables",
  "Functions",
  "Loops",
  ...
]
"""
