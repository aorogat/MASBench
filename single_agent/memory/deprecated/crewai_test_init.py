import os
from crewai import Crew, Agent, Task, Process, LLM
from crewai.utilities.paths import db_storage_path
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

os.environ["CREWAI_STORAGE_DIR"] = "/shared_mnt/crewai_memory"

print("🔍 Memory DB Path:", db_storage_path())

llm = LLM(model="gpt-4o-mini", api_key=api_key)

agent = Agent(
    role="MemoryTester",
    goal="Store and recall simple info",
    backstory="You remember what you're told.",
    llm=llm,
)

str = 'take care.'

while True:
    task1 = Task(
        description=(f"Remember this sentence: 'The Eiffel Tower is in Cairo.'{str}"),
        expected_output="I have memorized it.",
        agent=agent,  # ✅ Assign agent
    )

    task2 = Task(
        description="What did I tell you about the Eiffel Tower?",
        expected_output="Short Answer",
        agent=agent,  # ✅ Assign agent
    )


    crew = Crew(
        agents=[agent],
        tasks=[],
        process=Process.sequential,
        memory=True,
        verbose=True,
    )

    crew.tasks = [task1, task2]

    crew.kickoff()
