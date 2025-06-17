# streamlit_app.py

import streamlit as st
import os
from dotenv import load_dotenv
from agents.performance_analyzer import analyze_performance
from agents.expectation_retriever import retrieve_expectations
from agents.gap_explainer import explain_gap
from agents.resource_recommender import recommend_resources
from rag.rag_index import build_index

# Load API keys
load_dotenv()

# Build index if not already present
if not os.path.exists("rag_index.faiss"):
    build_index()

# App UI
st.set_page_config(page_title="Skill Gap Analyzer", layout="centered")
st.title("🎯 AI Skill Gap Analyzer ")

# Utility to clean and format Gemini output
def format_response_for_display(response):
    try:
        # If using LangChain GoogleGenerativeAI wrapper, response is likely a `BaseMessage` object
        if hasattr(response, 'content'):
            return response.content
        elif isinstance(response, dict) and "content" in response:
            return response["content"]
        elif isinstance(response, str):
            return response
        else:
            return str(response)
    except Exception as e:
        return f"❌ Error formatting response: {e}"

# Input form
with st.form("analyze_form"):
    st.subheader("📊 Enter Learner Performance")
    results_input = st.text_area(
        "Enter learner scores or feedback",
        value='{"scores": [{"topic": "DSA", "score": 45}, {"topic": "Array", "score": 40}, {"topic": "Machine Learning", "score": 30}]}'
    )

    st.subheader("🏢 Role Details")
    role = st.text_input("Role", "Backend Engineer")
    level = st.text_input("Level", "L3+")

    submitted = st.form_submit_button("Analyze Skill Gap")

if submitted:
    with st.spinner("🔍 Analyzing performance..."):
        insights = analyze_performance(results_input)
        formatted_insights = format_response_for_display(insights)

    st.success("✅ Performance Analysis Complete")
    with st.expander("🔍 Weak Areas"):
        st.markdown(formatted_insights, unsafe_allow_html=True)

    with st.spinner("📡 Retrieving industry expectations..."):
        expectations = retrieve_expectations(role, level)
        formatted_expectations = format_response_for_display(expectations)

    st.success("✅ Expectations Retrieved")
    with st.expander("💼 Industry Role Expectations"):
        st.markdown(formatted_expectations, unsafe_allow_html=True)

    with st.spinner("📉 Explaining the gap..."):
        gap = explain_gap(insights, expectations)
        formatted_gap = format_response_for_display(gap)

    st.success("✅ Gap Explanation Ready")
    with st.expander("📉 Skill Gap Explanation"):
        st.markdown(formatted_gap, unsafe_allow_html=True)

    with st.spinner("📚 Finding resources..."):
        resources = recommend_resources(gap)
        formatted_resources = format_response_for_display(resources)

    st.success("✅ Resources Recommended")
    with st.expander("🎓 Curated Learning Resources"):
        st.markdown(formatted_resources, unsafe_allow_html=True)
