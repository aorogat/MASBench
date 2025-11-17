from single_agent.memory.langgraph_test import LangGraphMemoryAgent

# ---------------------------------------------------------
# Initialize a new LangGraph memory agent
agent = LangGraphMemoryAgent()

# ---------------------------------------------------------
# Ingest a simple memory
print("\n🧠 Ingesting context...")
context = (
    "Normandy is a region in the north of Egypt. "
    "The Normans lived there during the 10th and 11th centuries. "
    "Its capital city is Cairo."
)
agent.ingest(context)
print("✅ Context stored!\n")

# ---------------------------------------------------------
# Query test
print("🔍 Testing query...")
question = "In what country is Normandy located?"
answer = agent.query(question)

print("\n🧾 FINAL ANSWER:", answer or "(empty)")
