from utils.llm import generate_research_questions
from utils.search import search_web

class ResearchAgent:
    def __init__(self, topic):
        self.topic = topic
        self.questions = []
        self.data = {}

    def plan(self):
        print("\n[1] Generating research questions...")
        self.questions = generate_research_questions(self.topic)

    def act(self):
        print("\n[2] Gathering information from the web...")
        for question in self.questions:
            print(f"🔍 Searching: {question}")
            self.data[question] = search_web(question)

    def compile_report(self):
        print("\n[3] Compiling report...")
        report = f"# Research Report on: {self.topic}\n\n"
        report += "## Introduction\nThis report explores the topic using AI-generated questions and web-based research.\n\n"

        for i, question in enumerate(self.questions, 1):
            report += f"### {i}. {question}\n"
            for result in self.data.get(question, []):
                report += f"**{result['title']}**\n\n{result['content']}\n\n"

        report += "## Conclusion\nThis report was generated using a ReAct-based agent (reasoning via Gemini + action via web search).\n"
        return report
