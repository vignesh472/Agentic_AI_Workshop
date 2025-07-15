Here's a detailed `README.md` for your **Multi-Agent RAG System** built with Streamlit, LangGraph, Gemini, and FAISS:

---

```markdown
# 🧠 Multi-Agent RAG System (LangGraph + Web + RAG + LLM)

This is an intelligent, fully agentic research assistant built using:

- 🧱 [LangGraph](https://github.com/langchain-ai/langgraph) for dynamic agent workflows  
- 🔍 DuckDuckGo Search for real-time web querying  
- 📄 RAG (Retrieval-Augmented Generation) using FAISS + Gemini embeddings  
- 🤖 Google Gemini 1.5 Flash for LLM processing and summarization  
- 💻 A simple Streamlit UI for local document queries and live search

---

## 🚀 Features

- **Multi-agent routing** (`web`, `rag`, `llm`) based on user query
- **Local document ingestion**: PDF, TXT, DOCX
- **Semantic chunking** with overlap using LangChain text splitter
- **FAISS vector database** for fast similarity-based retrieval
- **Query summarization** to generate concise results
- **Web search** fallback using DuckDuckGo
- **Gemini 1.5 Flash LLM** for cost-effective, high-speed reasoning

---

## 📁 Folder Structure

```

my\_docs/          # Put your local PDF/TXT/DOCX files here
app.py            # Main Streamlit application
README.md         # This file

````

---

## 🛠 Requirements

Install dependencies using pip:

```bash
pip install -r requirements.txt
````

> **Sample `requirements.txt`**

```txt
streamlit
pdfplumber
python-docx
langchain
langgraph
faiss-cpu
duckduckgo-search
python-dotenv
google-generativeai
```

---

## 🔑 Environment Setup

Create a `.env` file in your project root with your Gemini API key:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

You can get this key from: [https://makersuite.google.com/app](https://makersuite.google.com/app)

---

## 🏃 Running the App

```bash
streamlit run app.py
```

Once started, the app will:

1. Load documents from `my_docs/` (if available)
2. Split and embed content using Gemini Embeddings
3. Initialize a FAISS vectorstore for RAG
4. Wait for user input
5. Dynamically route the query to:

   * 🌐 Web Search
   * 📄 RAG (local document QA)
   * 🤖 Direct LLM response
6. Output a clean summarized response

---

## 🧠 How the Agent Workflow Works

The app uses **LangGraph** to define a modular `StateGraph` workflow with the following nodes:

* `router`: Classifies the query to determine the best route (`web`, `rag`, `llm`)
* `web`: Performs live DuckDuckGo search
* `rag`: Retrieves and answers using local documents via FAISS
* `llm`: Uses Gemini 1.5 directly
* `summarizer`: Summarizes the final content using Gemini

The conditional logic routes queries smartly based on context, maximizing relevance and minimizing cost.

---

## ✨ Example Use Cases

* "What is LangGraph?"
* "Summarize my research notes from this PDF"
* "What are the recent advancements in quantum AI?"
* "Can you summarize this 10-page DOCX document?"

---

## ❓ Troubleshooting

* **No API key found**: Ensure `.env` file exists and `GOOGLE_API_KEY` is defined.
* **No documents loaded**: Ensure `my_docs/` folder exists and contains readable files.
* **Web search failed**: Make sure you have internet access and the `duckduckgo-search` package is installed.

---

## 🧪 Future Improvements

* Memory and history tracking for multi-turn conversations
* UI enhancements: file uploader, chat-like interface
* Support for image and audio inputs
* More robust document parsing for scanned PDFs

---

## 👤 Author

Developed by **Vigneshwaran A**, powered by LangGraph, Gemini, and LangChain.

---
