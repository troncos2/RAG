import os
import wikipedia
from dotenv import load_dotenv

# 1. Load the env file
load_dotenv()

# 2. Get the agent from the env file
my_agent = os.getenv("USER_AGENT")
print(f"Using User-Agent: {my_agent}")

# 3. Set it in the library
wikipedia.set_user_agent(my_agent)

# 4. Try a direct search (bypassing LangChain completely)
try:
    print("Attempting to contact Wikipedia...")
    results = wikipedia.search("human evolution")
    print("\nAccessed wiki. Found articles:")
    print(results)
except Exception as e:
    print(f"\n Failed access: {e}")