from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.emotion_classifier import detect_emotion
from src.llm_response import generate_response
from src.retrieval import retrieve_similar
from src.welcome import generate_welcome
from src.closing import generate_closing
from src.conversation_end import is_conversation_end
from src.memory import ConversationMemory

app = FastAPI(title="Sentio API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    print("\n")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("🚀 Sentio API is ready")
    print("Open in browser:")
    print("👉 http://127.0.0.1:8000/docs")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("\n")

# Global memory for demo (frontend can manage sessions later)
memory = ConversationMemory()


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str
    emotion: str


@app.get("/")
def root():
    return {"message": "Sentio API running"}


@app.get("/welcome")
def welcome():
    msg = generate_welcome()
    return {"message": msg}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    user_query = request.message

    # detect emotion
    emotion, confidence = detect_emotion(user_query)

    # retrieve examples
    retrieved = retrieve_similar(user_query, emotion=emotion)

    # generate response
    response = generate_response(user_query, emotion, retrieved, memory)

    # store conversation
    memory.add(user_query, response)

    return {
        "reply": response,
        "emotion": emotion
    }


@app.post("/end")
def end_chat():

    closing = generate_closing()

    return {
        "message": closing
    }