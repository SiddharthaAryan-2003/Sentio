from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("❌ GROQ_API_KEY not found")

client = Groq(api_key=api_key)

from src.llm_response import client

def generate_welcome():

    prompt = """
    Generate a warm and inviting greeting for a supportive emotional AI assistant called Sentio.

    Rules:
    - 1 sentence only
    - warm and calm tone
    - supportive but not overly dramatic
    - avoid clichés like "How can I help you today?"
    - sound natural and human
    """

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You generate warm greetings for emotional support conversations."},
            {"role": "user", "content": prompt}
        ],
        temperature=1.0
    )

    return completion.choices[0].message.content.strip()