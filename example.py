import pandas as pd
from personality_engine import PersonalityEngine

# Initialize engine
engine = PersonalityEngine()

# Load your conversations CSV
df = pd.read_csv("f3293872.csv")

# Extract user memory (one-time operation)
memory = engine.extract_memory(df)
memory.to_json("user_memory.json")

# Transform responses based on personality
query = "How do I optimize this Python code?"
response = "Check time complexity. Look for parallelizable loops. Consider caching."

transformed = engine.transform(query, response)

print(f"Original: {transformed.original_response}")
print(f"Transformed: {transformed.transformed_response}")
print(f"Personality: {transformed.personality_used.value}")
print(f"Reasoning: {transformed.reasoning}")
