def get_rag_summary_prompt(concept, docs):
    # Combine the top-k retrieved chunks into a single text block
    combined = "\n---\n".join([doc.page_content for doc in docs])

    return f"""
You are an expert AI tutor helping learners understand difficult concepts.

The learner is struggling with the concept: **{concept}**

Below are internal knowledge base materials related to that concept:

--- START OF CONTENT ---
{combined}
--- END OF CONTENT ---

Task:
1. Summarize the core explanation clearly.
2. Provide a simple example or analogy.
3. Indicate how many unique sources were used.

Return your response in this JSON format:
{{
  "concept": "{concept}",
  "summary": "...",
  "example": "...",
  "source_count": {len(docs)}
}}
"""
