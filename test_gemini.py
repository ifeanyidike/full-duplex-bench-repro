import os
from dotenv import load_dotenv
from google import genai

load_dotenv("v1_v1.5/.env")

client = genai.Client(vertexai=False, api_key=os.getenv("GEMINI_API_KEY"))
print("Client created successfully")
print("Testing model access: gemini-3.1-flash-live-preview")

# Just list available models to verify key works
for m in client.models.list():
    if "live" in m.name.lower() or "flash" in m.name.lower():
        print(f"  {m.name}")