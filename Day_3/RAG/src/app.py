import streamlit as st
from preprocessing import load_and_chunk_pdfs, chunk_documents
from retrieval import HybridRetriever
from generation import generate_answer
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check for API key
if not os.getenv("GEMINI_API_KEY"):
    st.error("Gemini API key not found. Please create a .env file with GEMINI_API_KEY")
    st.stop()

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'retriever' not in st.session_state:
    st.session_state.retriever = None

# App title
st.title("📄 AI Research Papers QA System (Gemini)")
st.markdown("Ask questions about RAG research papers")

# Fixed values instead of sidebar input
chunk_size = 500
top_k = 3

# Document processing
if not st.session_state.processed:
    with st.spinner("Processing documents..."):
        try:
            raw_docs = load_and_chunk_pdfs("data")
            chunks = chunk_documents(raw_docs, chunk_size=chunk_size)
            retriever = HybridRetriever(chunks)
            st.session_state.retriever = retriever
            st.session_state.processed = True
            st.success(f"Processed {len(chunks)} document chunks from {len(raw_docs)} pages!")
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
            st.stop()

# Question input
question = st.text_input("Ask your question:", placeholder="e.g. What is multi-head attention?")

# Sample questions
sample_qs = [
    "Larger models make increasingly efficient use of in-context information?",
    "what are the  Applications of Attention in our Model?",
    "Aggregate performance for all 42 accuracy-denominated benchmarks",
    "Why Self-Attention",
    "what is LAMBADA"
]
st.caption("Sample questions: " + " | ".join([f"`{q}`" for q in sample_qs]))

# Handle question submission
if question and st.session_state.retriever:
    start_time = time.time()
    
    try:
        with st.spinner("Searching documents..."):
            context_chunks = st.session_state.retriever.retrieve(question)
        
        with st.spinner("Generating answer with Gemini..."):
            answer, sources = generate_answer(question, context_chunks)
            elapsed = time.time() - start_time
        
        # Display results
        st.subheader("Answer:")
        st.info(answer)
        
        st.subheader("Sources:")
        for source in sources:
            st.write(f"- {source}")
        
        st.caption(f"Response time: {elapsed:.2f} seconds")
        
        # Show context chunks
        st.subheader("Relevant Contexts:")
        for i, (text, filename, page) in enumerate(context_chunks):
            st.markdown(f"**Context {i+1} from {filename} (Page {page}):**")
            st.text(text[:500] + "..." if len(text) > 500 else text)
            st.divider()
            
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
