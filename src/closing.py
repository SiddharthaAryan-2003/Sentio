from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("❌ GROQ_API_KEY not found")

client = Groq(api_key=api_key)

from src.llm_response import client

def generate_closing():

    prompt = """
Generate a short supportive closing message for someone finishing an emotional support conversation.

Rules:
- 1 sentence
- calm tone
- encouraging but not dramatic
"""

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You generate kind closing messages for emotional support chats."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.9
    )

    return completion.choices[0].message.content.strip()