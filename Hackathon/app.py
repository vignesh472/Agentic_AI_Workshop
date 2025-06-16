import streamlit as st
import json
import pandas as pd
import os
import re
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from retriever.retriever_loader import load_retriever
import google.generativeai as genai

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("models/gemini-1.5-flash")

# Extract valid problem/video links
def extract_links(text: str) -> dict:
    problems_cleaned = list(set(re.findall(r"https?://[^\s\"\'\)\]]+", text)))
    problems_filtered = [link for link in problems_cleaned if any(p in link for p in ["leetcode", "geeksforgeeks", "hackerrank"])]
    videos_filtered = [link for link in problems_cleaned if "youtube.com/watch" in link]
    return {"problems": problems_filtered[:3], "videos": videos_filtered[:2]}

# Generate quiz questions
def generate_quiz(topic: str, num_questions=5) -> list:
    prompt = f"""
    Generate exactly {num_questions} multiple-choice quiz questions about {topic} for data structures and algorithms.
    Each question must have:
    - A clear question text
    - Exactly 4 options
    - The correct answer index (0-3)
    
    Return ONLY valid JSON in this exact format:
    {{
        "questions": [
            {{
                "question": "What is the time complexity of binary search?",
                "options": ["O(n)", "O(nlogn)", "O(logn)", "O(1)"],
                "answer": 2
            }}
        ]
    }}
    Only return the JSON. No extra text.
    """
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()
        data = json.loads(text)
        return data.get("questions", [])
    except json.JSONDecodeError:
        st.error("Failed to parse quiz questions. The AI returned invalid JSON format.")
        return []
    except Exception as e:
        st.error(f"Error generating quiz: {str(e)}")
        return []

# Fallback feedback if no context
def fallback_feedback(topic: str) -> str:
    fallback_prompt = f"""
    Explain why the topic "{topic}" is important for technical interviews. 
    List 2-3 core subtypes or patterns and common mistakes.
    """
    try:
        return model.generate_content(fallback_prompt).text.strip()
    except:
        return "Gemini could not generate feedback."

# Generate resources
def generate_resources(topic: str) -> dict:
    prompt = f"""
    For the topic "{topic}", return:
    {{
        "problems": ["https://leetcode.com/... (3 max)"],
        "videos": ["https://youtube.com/... (2 max)"]
    }}
    Do NOT include text or explanation, only valid JSON with real URLs.
    """
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        try:
            data = json.loads(text)
            return extract_links(json.dumps(data)) if isinstance(data, dict) else extract_links(text)
        except:
            return extract_links(text)
    except:
        return None

# Suggest more topics
def suggest_more_topics(weak_topics: list) -> list:
    suggestion_prompt = f"""
    A student is weak in the following DSA topics: {', '.join(weak_topics)}.
    Suggest 2-3 additional topics they should study to strengthen foundational skills.
    Return only a comma-separated list, no explanation.
    """
    try:
        suggestions = model.generate_content(suggestion_prompt).text.strip()
        return [s.strip() for s in suggestions.split(",") if s.strip()]
    except:
        return []

# App UI
st.set_page_config(page_title="DSA Feedback Agent", layout="wide")
st.title("🧠 UpSkiller")

# Session state
if 'quiz_data' not in st.session_state:
    st.session_state.quiz_data = None
if 'user_answers' not in st.session_state:
    st.session_state.user_answers = {}
if 'quiz_submitted' not in st.session_state:
    st.session_state.quiz_submitted = False
if 'weak_topics' not in st.session_state:
    st.session_state.weak_topics = []
if 'show_quiz' not in st.session_state:
    st.session_state.show_quiz = False

# Always assessment mode
st.subheader("📝 Create Your Assessment")
topics_input = st.text_input("Enter topics (comma separated):", placeholder="e.g. Arrays, Trees, Graphs")

topics = [t.strip() for t in topics_input.split(",") if t.strip()]

if st.button("Generate Quiz") and topics:
    with st.spinner("Generating quiz questions..."):
        all_questions = []
        for topic in topics:
            questions = generate_quiz(topic)
            if questions:
                all_questions.extend(questions)
        
        if all_questions:
            st.session_state.quiz_data = all_questions[:5]
            st.session_state.user_answers = {i: None for i in range(len(st.session_state.quiz_data))}
            st.session_state.quiz_submitted = False
            st.session_state.show_quiz = True
            st.success("Quiz generated successfully!")
        else:
            st.error("Failed to generate quiz questions. Please try again.")

# Quiz Display
if st.session_state.show_quiz and st.session_state.quiz_data:
    st.subheader("📝 Quiz")
    with st.form("quiz_form"):
        for i, question in enumerate(st.session_state.quiz_data):
            st.markdown(f"**Q{i+1}: {question['question']}**")
            options = question.get("options", ["Option 1", "Option 2", "Option 3", "Option 4"])
            st.session_state.user_answers[i] = st.radio(
                f"Select an answer for Q{i+1}:",
                options,
                key=f"q_{i}",
                index=None
            )
            st.write("---")
        
        if st.form_submit_button("Submit Quiz"):
            if None in st.session_state.user_answers.values():
                st.warning("Please answer all questions before submitting!")
            else:
                st.session_state.quiz_submitted = True
                
                # Calculate score
                correct = 0
                weak_topics = set()
                for i, question in enumerate(st.session_state.quiz_data):
                    correct_answer = question.get("answer", 0)
                    if st.session_state.user_answers[i] == question["options"][correct_answer]:
                        correct += 1
                    else:
                        topic_keywords = {
                            "array": "Arrays",
                            "linked list": "Linked Lists",
                            "tree": "Trees",
                            "sort": "Sorting",
                            "graph": "Graphs",
                            "dynamic programming": "DP"
                        }
                        for keyword, topic_name in topic_keywords.items():
                            if keyword in question["question"].lower():
                                weak_topics.add(topic_name)
                                break
                        else:
                            weak_topics.add("General DSA")
                
                score = (correct / len(st.session_state.quiz_data)) * 100
                st.session_state.weak_topics = list(weak_topics)
                
                st.subheader("📊 Your Results")
                st.metric("Overall Score", f"{score:.1f}%")
                
                if score < 60 and st.session_state.weak_topics:
                    st.warning(f"⚠️ Weak areas detected: {', '.join(st.session_state.weak_topics)}")
                else:
                    st.success("✅ Good performance across all topics!")

# Show feedback
if st.session_state.quiz_submitted and st.session_state.weak_topics:
    retriever = load_retriever()
    
    for topic in st.session_state.weak_topics:
        with st.expander(f"🔍 Feedback & Resources for: {topic}", expanded=False):
            with st.spinner("Generating feedback..."):
                docs = retriever.get_relevant_documents(
                    f"Why is {topic} important in interviews? What are its subtypes?"
                )

                if docs:
                    st.caption("🧠 Source: Retrieved industry context")
                    context = "\n".join([doc.page_content for doc in docs])
                    prompt = f"""
                    Based on the following context, explain the topic "{topic}" in the context of coding interviews.
                    Highlight its significance and 2-3 main patterns.

                    {context}
                    """
                    try:
                        feedback = model.generate_content(prompt).text.strip()
                    except:
                        feedback = fallback_feedback(topic)
                else:
                    st.caption("💡 No RAG data found - fallback to Gemini's general knowledge.")
                    feedback = fallback_feedback(topic)

                st.markdown(f"**📝 Feedback:**\n\n{feedback}")

            st.markdown("---")
            st.markdown("**📚 Practice Resources**")

            resources = generate_resources(topic)
            if resources and (resources["problems"] or resources["videos"]):
                if resources["problems"]:
                    st.markdown("✅ **Problems:**")
                    for link in resources["problems"]:
                        st.markdown(f"- [{link.split('//')[1].split('/')[0]}]({link})")

                if resources["videos"]:
                    st.markdown("🎥 **Videos:**")
                    for link in resources["videos"]:
                        st.markdown(f"- [YouTube]({link})")
            else:
                st.warning("No valid resources generated.")

    # More topics
    st.markdown("---")
    st.subheader("🧭 More Topics to Study")
    related = suggest_more_topics(st.session_state.weak_topics)
    if related:
        st.markdown("Based on your weaknesses, you might want to also review:")
        for topic in related:
            st.markdown(f"- 🔁 {topic}")
    else:
        st.info("No additional topics could be generated.")
