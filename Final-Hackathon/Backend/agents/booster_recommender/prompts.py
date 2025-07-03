def get_booster_prompt(concepts, preference, category, assessment_result):
    concept_details = ""
    for concept in concepts:
        subtopics = category.get(concept, [])
        concept_details += f"- {concept}: {', '.join(subtopics)}\n"

    assessment_summary = ""
    for concept in concepts:
        if assessment_result.get("concept") == concept:
            scores = assessment_result.get("scores", {})
            for idx, val in scores.items():
                score_obj = val.get("score", {})
                feedback = val.get("feedback", "")
                assessment_summary += (
                    f"\nAssessment for {concept} (Q{idx}):\n"
                    f"- Correctness: {score_obj.get('correctness')}\n"
                    f"- Completeness: {score_obj.get('completeness')}\n"
                    f"- Reasoning: {score_obj.get('reasoning')}\n"
                    f"- Feedback: {feedback.strip()}\n"
                )
        # print("assessment_summary", assessment_summary)
    return f"""
You are an expert AI tutor designing **targeted booster content** to help a learner master difficult topics.

Context:
- Learner's preferred format: **{preference}**
- Concepts to focus on:
{concept_details.strip()}

Assessment insights:
{assessment_summary.strip()}

🧠 Task:
Recommend only **3 to 4 concise boosters** that collectively:
1. Address all the concepts and subtopics listed
2. Directly respond to the feedback and weak scores observed in the assessment
3. Match the learner’s interactive style
4. Are focused, unique, and under 15 minutes each

✅ Return format (strict valid JSON list):

[
  {{
    "title": "Fix the Flow of Recursion",
    "booster_type": "video" or "coding challenge" (choose one per booster),
    "concepts_covered": ["Recursion"],
    "subtopics": ["Base Case", "Call Stack"],
    "format": "interactive",
    "description": "Interactive visualization showing how recursion works, with a drag-and-drop activity on base cases.",
    "estimated_duration": "12 minutes"
  }},
  ...
]
"""
