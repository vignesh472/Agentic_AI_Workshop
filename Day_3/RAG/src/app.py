import streamlit as st
from preprocessing import load_and_chunk_pdfs, chunk_documents
from retrieval import HybridRetriever
from generation import generate_answer  # Should return: answer, sources
import time
import os
from dotenv import load_dotenv
from opik import Opik

# Load environment variables
load_dotenv()

# Check for Gemini API key
if not os.getenv("GEMINI_API_KEY"):
    st.error("❌ Gemini API key not found. Please create a .env file with GEMINI_API_KEY")
    st.stop()

# Initialize Opik client (no log_retrieval or log_generation — we use traces)
opik = Opik(api_key="CmpitYp2tAdVUq0m86b791c2F", project_name="RAG QA System")

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'retriever' not in st.session_state:
    st.session_state.retriever = None

# App header
st.title("📄 AI Research Papers QA System (Gemini)")
st.markdown("Ask questions about RAG research papers")

chunk_size = 500
top_k = 3

# Document processing
if not st.session_state.processed:
    with st.spinner("Processing documents..."):
        try:
            raw_docs = load_and_chunk_pdfs("../data")  # Adjust path if needed
            chunks = chunk_documents(raw_docs, chunk_size=chunk_size)
            retriever = HybridRetriever(chunks)
            st.session_state.retriever = retriever
            st.session_state.chunks = chunks
            st.session_state.processed = True
            st.success(f"✅ Processed {len(chunks)} chunks from {len(raw_docs)} documents.")
        except Exception as e:
            st.error(f"❌ Error processing documents: {str(e)}")
            st.stop()

# Question input
question = st.text_input("Ask your question:", placeholder="e.g. What is multi-head attention?")

# Sample suggestions
sample_qs = [
    "Larger models make increasingly efficient use of in-context information?",
    "What are the applications of Attention in our model?",
    "Aggregate performance for all 42 accuracy-denominated benchmarks",
    "Why Self-Attention?",
    "What is LAMBADA?"
]
st.caption("💡 Sample questions: " + " | ".join([f"`{q}`" for q in sample_qs]))

# Main QA handling
if question and st.session_state.retriever:
    start_time = time.time()
    try:
        with st.spinner("🔍 Retrieving relevant contexts..."):
            context_chunks = st.session_state.retriever.retrieve(question)
            opik.trace(
                name="retrieval",
                input={"query": question},
                output={"contexts": [text for text, _, _ in context_chunks]},
                metadata={"type": "document_retrieval"}
            ).end()

        with st.spinner("💡 Generating answer using Gemini..."):
            # Only expecting 2 values: answer and sources
            answer, sources, token_usage = generate_answer(question, context_chunks)
            token_usage = {}  # Gemini doesn't return this directly
            elapsed = time.time() - start_time

            opik.trace(
                name="generation",
                input={"query": question, "contexts": [text for text, _, _ in context_chunks]},
                output={"answer": answer, "sources": sources},
                metadata={
                    "model": "gemini-2.5-flash",
                    "latency_ms": int(elapsed * 1000),
                    "tokens": token_usage
                }
            ).end()

        # Displaying results
        st.subheader("Answer:")
        st.info(answer)

        st.subheader("Sources:")
        for source in sources:
            st.write(f"- {source}")

        st.caption(f"⏱️ Response time: {elapsed:.2f} seconds")

        st.subheader("Relevant Contexts:")
        for i, (text, filename, page) in enumerate(context_chunks):
            st.markdown(f"**Context {i+1} from {filename} (Page {page}):**")
            st.text(text[:500] + "..." if len(text) > 500 else text)
            st.divider()

    except Exception as e:
        st.error(f"❌ Error generating answer: {str(e)}")
