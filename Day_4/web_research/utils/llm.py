import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")

def generate_research_questions(topic):
    prompt = f"Generate 5–6 insightful research questions about the topic: '{topic}'"
    response = model.generate_content(prompt)
    
    lines = response.text.strip().split("\n")
    questions = [line.strip("0123456789. ").strip() for line in lines if line.strip()]
    return questions
