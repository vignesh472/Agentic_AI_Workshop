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

# Build index if not already
if not os.path.exists("rag_index.faiss"):
    build_index()

# UI Title
st.set_page_config(page_title="Skill Gap Analyzer", layout="centered")
st.title("🎯 AI Skill Gap Analyzer (Gemini 1.5 Flash)")

# Input section
with st.form("analyze_form"):
    st.subheader("📊 Enter Learner Performance")
    results_input = st.text_area("Enter learner scores or feedback", 
        value='{"scores": [{"topic": "DSA", "score": 45}, {"topic": "Array", "score": 40}, {"topic": "Machine Learning", "score": 30}]}')

    st.subheader("🏢 Role Details")
    role = st.text_input("Role", "Backend Engineer")
    level = st.text_input("Level", "L3+")

    submitted = st.form_submit_button("Analyze Skill Gap")

if submitted:
    with st.spinner("Analyzing performance..."):
        insights = analyze_performance(results_input)
    st.success("Performance Analysis Complete")
    st.markdown("### 🔍 Weak Areas")
    st.write(insights)

    with st.spinner("Retrieving industry expectations..."):
        expectations = retrieve_expectations(role, level)
    st.success("Expectations Retrieved")
    st.markdown("### 💼 Industry Role Expectations")
    st.write(expectations)

    with st.spinner("Explaining the gap..."):
        gap = explain_gap(insights, expectations)
    st.success("Gap Explanation Ready")
    st.markdown("### 📉 Skill Gap Explanation")
    st.write(gap)

    with st.spinner("Finding resources..."):
        resources = recommend_resources(gap)
    st.success("Resources Recommended")
    st.markdown("### 🎓 Curated Learning Resources")
    st.write(resources)
