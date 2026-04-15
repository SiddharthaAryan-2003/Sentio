# Sentio: Emotion-Aware Support Chatbot

Sentio is an emotion-aware, retrieval-augmented conversational assistant for supportive (non-clinical) mental well-being conversations.

It combines:
- emotion classification,
- semantic retrieval from a curated support dataset,
- short-term conversation memory, and
- Groq-hosted LLM generation

to produce empathetic, context-aware responses.

## Important Disclaimer

This project is **not** a medical tool and does **not** provide diagnosis or treatment.

If someone is in immediate danger or crisis, contact local emergency services or a professional crisis line.

## Features

- Emotion detection using `SamLowe/roberta-base-go_emotions`
- Retrieval using sentence embeddings (`all-MiniLM-L6-v2`) + FAISS
- Context-aware response generation through Groq (`llama-3.1-8b-instant`)
- FastAPI backend with frontend-ready REST endpoints
- Static HTML/CSS/JS frontend
- In-memory conversation context for more coherent replies

## Project Structure

```text
project/
├── src/
│   ├── api.py
│   ├── ChatBot.py
│   ├── emotion_classifier.py
│   ├── retrieval.py
│   ├── llm_response.py
│   ├── welcome.py
│   ├── closing.py
│   ├── conversation_end.py
│   └── memory.py
├── frontend/
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── chat.html
│   └── styles.css
├── data/
│   └── emotional_dataset.csv
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+ (tested on 3.14)
- `pip`
- A valid Groq API key

## Setup

From project root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create `.env` in project root:

```env
GROQ_API_KEY=your_actual_groq_api_key
```

## Run the Backend API

From project root:

```bash
. .venv/bin/activate
HF_HOME="$PWD/.hf-cache" TRANSFORMERS_CACHE="$PWD/.hf-cache" uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```

Then open:
- FastAPI docs: `http://127.0.0.1:8000/docs`
- Health/root check: `http://127.0.0.1:8000/`

## Run the Frontend

From project root (recommended command):

```bash
python3 -m http.server 5500 --directory .
```

Open:
- `http://127.0.0.1:5500/frontend/index.html`

The frontend is configured to call backend API at:
- `http://127.0.0.1:8000`

## API Reference

### `GET /`

Returns backend status.

Example response:

```json
{
  "message": "Sentio API running"
}
```

### `GET /welcome`

Returns a generated welcome message.

Example response:

```json
{
  "message": "I'm glad you're here today; take your time and share whatever feels most important."
}
```

### `POST /chat`

Accepts user message and returns assistant reply + detected emotion.

Request body:

```json
{
  "message": "I feel anxious today"
}
```

Response body:

```json
{
  "reply": "That sounds really heavy to carry right now. Want to share what has been making today feel most anxious?",
  "emotion": "fear"
}
```

### `POST /end`

Returns a short closing message.

Example response:

```json
{
  "message": "I'm glad we talked today; take care of yourself and come back whenever you need support."
}
```

## Optional: Run CLI Chatbot

From project root:

```bash
. .venv/bin/activate
python -m src.ChatBot
```

## Troubleshooting

### 1) `Address already in use` on port 8000

Another server process is already using that port.

```bash
pkill -f "uvicorn src.api:app"
```

Then start backend again.

### 2) `GROQ_API_KEY not found`

- Ensure `.env` exists in project root.
- Ensure key name is exactly `GROQ_API_KEY`.
- Restart backend after editing `.env`.

### 3) Hugging Face cache/download permission errors

Use this backend command (already recommended) to keep cache local to project:

```bash
HF_HOME="$PWD/.hf-cache" TRANSFORMERS_CACHE="$PWD/.hf-cache" uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```

### 4) Frontend loads but chat fails

- Confirm backend is running on `127.0.0.1:8000`
- Check backend logs for auth/model errors
- Open browser dev tools network tab and inspect `POST /chat`

## Pre-Push Checklist (GitHub)

- `.env` is **not** committed
- API key is valid locally
- Backend starts cleanly
- Frontend opens and can send/receive messages
- `README.md` reflects current endpoints (`reply`, not `response`)

Suggested root `.gitignore` entries:

```gitignore
.env
.venv/
.hf-cache/
__pycache__/
*.pyc
```

## License

This project includes a `LICENSE` file in the repository root.