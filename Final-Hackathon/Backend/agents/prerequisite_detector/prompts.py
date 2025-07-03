def get_dynamic_prerequisite_prompt(weak_concepts: list):
    return f"""
You are an AI tutor specialized in computer science education.

The learner is struggling with the following concepts:
{weak_concepts}

Your task:
1. For each weak concept, identify the foundational or prerequisite concepts that must be understood first.
2. These prerequisites may include both direct and indirect dependencies.
3. Avoid repeating concepts across lists unnecessarily, but ensure each weak concept has its own relevant prerequisite list.
4. Focus strictly on computing, programming, and algorithmic reasoning.

📌 Return the result in the following JSON format:
{{
  "Loops": ["Variables", "Control Flow", "Functions"],
  "Recursion": ["Functions", "Call Stack", "Base Case"],
  ...
}}

Each key should be a weak concept, and its value should be an array of prerequisite concepts (gaps).
"""
