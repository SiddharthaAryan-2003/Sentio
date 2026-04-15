from groq import Groq
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Get API key
api_key = os.getenv("GROQ_API_KEY")

# Safety check
if not api_key:
    raise ValueError("❌ GROQ_API_KEY not found. Check your .env file.")

# Initialize client
client = Groq(api_key=api_key)


def generate_response(user_query, emotion, retrieved_examples, memory=None):

    # -------- Build retrieved examples context --------
    examples_context = ""

    if retrieved_examples is not None and len(retrieved_examples) > 0:
        for _, row in retrieved_examples.iterrows():
            examples_context += f"""
Situation: {row['situation']}
User: {row['dialogue']}
Assistant: {row['response']}
"""
    else:
        examples_context = "No similar examples available."

    # -------- Build memory context --------
    memory_context = ""

    if memory:
        try:
            past = memory.get_context()
            if past.strip():
                memory_context = f"\nRecent conversation:\n{past}\n"
        except Exception:
            memory_context = ""

    # -------- Final prompt --------
    prompt = f"""
You are Sentio, an empathetic emotional support assistant.

Current user emotion: {emotion}

{memory_context}

Relevant examples:
{examples_context}

User message:
"{user_query}"

Instructions:
- Be warm, empathetic, and human-like
- Acknowledge emotions naturally
- Do NOT sound robotic or scripted
- Keep response concise (2–4 sentences)
- Do NOT repeat examples
- Try to uphold the conversation in such a way that the user exposes more about their feelings and situation, but do not be pushy
- Try to provide some new insight or perspective if possible, but keep it grounded and realistic
- Give the user any music recommendation once in a while if you think it can help them based on their current emotion and situation

"""

    # -------- LLM call --------
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a compassionate emotional support assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.8
    )

    return completion.choices[0].message.content.strip()