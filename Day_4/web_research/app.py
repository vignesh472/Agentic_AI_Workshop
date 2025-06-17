from agent.research_agent import ResearchAgent
import os

def main():
    topic = input("📝 Enter your research topic: ")
    agent = ResearchAgent(topic)

    agent.plan()
    agent.act()
    report = agent.compile_report()

    os.makedirs("output", exist_ok=True)
    with open("output/research_report.md", "w") as f:
        f.write(report)

    print("\n✅ Report generated successfully at: output/research_report.md")

if __name__ == "__main__":
    main()
