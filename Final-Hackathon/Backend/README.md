Here’s a complete and clean `README.md` for your **Agentic AI-Based Concept Mastery Evaluator and Prerequisite Recommender** project, tailored to your architecture, agents, backend setup, and dataset usage:

---

```markdown
# 🧠 Agentic AI-Based Concept Mastery Evaluator and Prerequisite Recommender

## 🚀 Objective
This system uses an Agentic AI architecture to dynamically evaluate learners’ mastery of concepts and intelligently recommend prerequisite refreshers. It ensures foundational understanding before progressing to advanced topics.

---

## 🧩 Core Features
- ✅ Concept-wise Mastery Evaluation
- 🔍 Automated Prerequisite Detection
- 🎯 Booster Recommendations (micro-lessons, exercises, quizzes)
- 📚 RAG-based Prerequisite Content Retrieval
- 📊 Knowledge Map & Progress Tracking (frontend-ready)
- 🔁 Continual Agent Learning Loop

---

## 🧠 Multi-Agent Architecture

### 1. **Mastery Evaluation Agent**
- **Inputs**: Quiz scores, retry count, coding logs, time spent
- **Output**: Mastery status per concept
- **Collection**: `mastery_logs`

---

### 2. **Prerequisite Gap Detector Agent**
- **Inputs**: Mastery report + Concept dependency tree
- **Output**: Missing prerequisites
- **Collection**: `prerequisite_gaps`

---

### 3. **Adaptive Assessment Generator Agent**
- **Inputs**: Concept + learner mastery
- **Output**: Dynamic questions with reasoning evaluation

---

### 4. **Booster Recommendation Agent**
- **Inputs**: Prerequisite gaps + learner profile
- **Output**: Micro-lessons, exercises, quizzes
- **Collection**: `boosters_assigned`

---

### 5. **RAG-Powered Prerequisite Retriever Agent**
- **Inputs**: Weak concepts
- **Output**: Top internal explanation chunks
- **Collection**: `retrieval_log`

---

## 🗂️ Project Structure

```

Backend/
├── agents/
│   ├── mastery\_evaluator/
│   │   └── agent.py
│   ├── prerequisite\_detector/
│   │   └── agent.py
│   ├── adaptive\_assessment/
│   │   └── agent.py
│   ├── booster\_recommender/
│   │   └── agent.py
│   └── prerequisite\_retriever/
│       ├── agent.py
│       └── vectorstore.py
├── routes/
│   └── evaluate.py
├── models/
│   ├── mastery.py
│   ├── prerequisite.py
│   ├── booster.py
│   └── retrieval\_log.py
├── data/
│   └── (your .txt or .pdf internal resources for RAG)
├── config.py
├── main.py
├── requirements.txt
└── README.md

````

---

## 🛠️ Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/agentic-ai-mastery.git
cd agentic-ai-mastery/Backend
````

---

### 2. Create `.env` file

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=models/embedding-001
MONGO_URI=mongodb://localhost:27017
```

---

### 3. Install Requirements

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

### 4. Run the App

```bash
uvicorn main:app --host 0.0.0.0 --port 5000 --reload
```

Backend available at: [http://localhost:5000](http://localhost:5000)

---

## 📁 Dataset Setup for RAG

* Place `.txt` or extracted `.pdf` content in the `data/` folder.
* The Retriever Agent will load these files using `langchain` and embed them with `GoogleGenerativeAIEmbeddings`.

Example:

```
data/
├── functions.txt
├── recursion.txt
├── call_stack.txt
```

---

## 📬 API Endpoints

| Endpoint                              | Method | Description                      |
| ------------------------------------- | ------ | -------------------------------- |
| `/api/evaluate/mastery`               | POST   | Evaluate mastery for a user      |
| `/api/evaluate/gaps`                  | POST   | Detect prerequisite gaps         |
| `/api/evaluate/assess`                | POST   | Generate follow-up questions     |
| `/api/evaluate/boosters`              | POST   | Recommend micro-lessons          |
| `/api/evaluate/retrieve-prerequisite` | POST   | RAG fetch of supporting material |

---

## 🧪 Sample Request Payload

```json
{
  "user_id": "u123",
  "quiz_scores": { "Functions": 9, "Recursion": 6 },
  "retry_logs": { "Recursion": 3 },
  "time_spent": { "Functions": 12, "Recursion": 25 }
}
```

---

## ✅ Technologies Used

* **FastAPI** - backend framework
* **MongoDB** - data storage
* **LangChain** - agent orchestration
* **FAISS** - vector store for RAG
* **Google Generative AI** - embeddings and LLM
* **Pydantic** - schema validation

---

## 📈 Future Enhancements

* ✅ Interactive Concept Map UI
* 🔁 Feedback loop for booster effectiveness
* 📊 Mentor analytics dashboard
* 🧠 Personalized learning paths

---

## 📄 License

MIT License

---

## 👨‍💻 Maintainer

[Vigneshwaran](https://github.com/vigneshwaran-ai)

```

---

Let me know if you'd like this as a downloadable file or integrated with your existing FastAPI docs (`/docs`).
```
