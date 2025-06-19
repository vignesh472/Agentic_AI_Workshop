import streamlit as st
import os
from dotenv import load_dotenv

from agents.performance_analyzer import analyze_performance
from agents.expectation_retriever import retrieve_expectations
from agents.gap_explainer import explain_gap
from agents.resource_recommender import recommend_resources
from rag.rag_index import build_index

# Load environment variables
load_dotenv()

# Build FAISS index if not present
if not os.path.exists("rag_index.faiss/index.faiss"):
    with st.spinner("🔧 Building vector index from role expectations PDF..."):
        build_index()

# UI setup
st.set_page_config(page_title="AI Skill Gap Analyzer", layout="centered")
st.title("🎯 AI Skill Gap Analyzer")

# Response formatter
def format_response_for_display(response):
    try:
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
        value='{"scores": [{"topic": "DSA", "score": 45}, {"topic": "Array", "score": 40}, {"topic": "Machine Learning", "score": 30}]}',
        height=150
    )

    st.subheader("🏢 Role Details")
    role = st.text_input("Role", "Backend Engineer")
    level = st.text_input("Level", "L3+")

    submitted = st.form_submit_button("🔎 Analyze Skill Gap")

# Workflow
if submitted:
    # Step 1: Analyze performance
    with st.spinner("🔍 Analyzing learner performance..."):
        insights = analyze_performance(results_input)
        formatted_insights = format_response_for_display(insights)
    st.success("✅ Performance Analysis Complete")
    with st.expander("🧠 Weak Areas Identified"):
        st.markdown(formatted_insights, unsafe_allow_html=True)

    # Step 2: Retrieve expectations
    with st.spinner("📡 Retrieving expectations from benchmark data..."):
        expectations = retrieve_expectations(role, level)
        formatted_expectations = format_response_for_display(expectations)
    st.success("✅ Industry Expectations Fetched")
    with st.expander("💼 Role Expectations"):
        st.markdown(formatted_expectations, unsafe_allow_html=True)

    # Step 3: Explain skill gap
    with st.spinner("🧩 Explaining the skill gap..."):
        gap = explain_gap(insights, expectations)
        formatted_gap = format_response_for_display(gap)
    st.success("✅ Gap Explanation Generated")
    with st.expander("📉 Skill Gap Explanation"):
        st.markdown(formatted_gap, unsafe_allow_html=True)

    # Step 4: Recommend learning resources
    with st.spinner("📚 Recommending learning resources..."):
        resources = recommend_resources(gap)
        formatted_resources = format_response_for_display(resources)
    st.success("✅ Resources Suggested")
    with st.expander("🎓 Learning Resources"):
        st.markdown(formatted_resources, unsafe_allow_html=True)
