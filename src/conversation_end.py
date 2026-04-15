from src.llm_response import client

def is_conversation_end(user_message):

    prompt = f"""
    Determine if the following message indicates the user is ending the conversation.

    Message:
    "{user_message}"

    Respond ONLY with YES or NO.
    """

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You detect whether a conversation is ending."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    result = completion.choices[0].message.content.strip().upper()

    return result == "YES"