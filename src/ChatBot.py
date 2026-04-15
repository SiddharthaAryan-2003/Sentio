from src.emotion_classifier import detect_emotion
from src.llm_response import generate_response
from src.retrieval import retrieve_similar
from src.welcome import generate_welcome
from src.closing import generate_closing
from src.conversation_end import is_conversation_end
from src.memory import ConversationMemory

def run_chatbot():

    memory = ConversationMemory()

    print("\nSentio — Emotion Aware Support Bot\n")

    # Generative welcome
    welcome = generate_welcome()
    print(f"\nSentio: {welcome}\n")

    while True:

        user_query = input("You: ")

        if user_query.lower() == "exit":
            break

        # detect emotion
        emotion, confidence = detect_emotion(user_query)

        # retrieve examples
        retrieved = retrieve_similar(user_query, emotion=emotion)

        # generate response
        response = generate_response(user_query, emotion, retrieved, memory)

        print(f"\nSentio: {response}\n")

        # store memory
        memory.add(user_query, response)

        # detect conversation end
        if is_conversation_end(user_query):

            closing = generate_closing()

            print(f"\nSentio: {closing}\n")

            break


if __name__ == "__main__":
    run_chatbot()