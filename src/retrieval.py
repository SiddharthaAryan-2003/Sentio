import os
import sys
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Get project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Load dataset
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "emotional_dataset.csv")
df = pd.read_csv(DATA_PATH)

# Drop useless columns
df = df.drop(columns=["Unnamed: 5", "Unnamed: 6"], errors="ignore")

# Rename columns
df = df.rename(columns={
    "Situation": "situation",
    "emotion": "emotion",
    "empathetic_dialogues": "dialogue",
    "labels": "response"
})

# Clean dialogue text
df["dialogue"] = (
    df["dialogue"]
    .str.replace("Customer:", "", regex=False)
    .str.replace("Agent:", "", regex=False)
    .str.replace("\n", " ", regex=False)
    .str.strip()
)

# Remove null rows
df = df.dropna()

# Emotion mapping (10-core emotional taxonomy)
EMOTION_MAP = {
    # Joy
    "joyful": "joy", "happy": "joy", "excited": "joy",
    "content": "joy", "impressed": "joy",

    # Sadness
    "sad": "sadness", "lonely": "sadness",
    "nostalgic": "sadness", "disappointed": "sadness",

    # Anger
    "angry": "anger", "furious": "anger", "annoyed": "anger",

    # Fear
    "afraid": "fear", "anxious": "fear", "terrified": "fear",
    "stressed": "fear", "stress": "fear",

    # Love / Attachment
    "sentimental": "love", "grateful": "love", "affectionate": "love",

    # Surprise
    "surprised": "surprise", "shocked": "surprise",

    # Disgust
    "disgusted": "disgust",

    # Guilt / Shame
    "guilty": "guilt", "ashamed": "guilt",

    # Pride / Confidence
    "proud": "pride", "confident": "pride",

    # Hope / Anticipation
    "hopeful": "hope", "anticipating": "hope", "prepared": "hope"
}

# Normalize emotions
df["emotion_normalized"] = df["emotion"].map(EMOTION_MAP)
df = df.dropna(subset=["emotion_normalized"])

# Create RAG text
df["rag_text"] = (
    "Situation: " + df["situation"] +
    " | Emotion: " + df["emotion_normalized"] +
    " | Dialogue: " + df["dialogue"]
)

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings for RAG text
texts = df["rag_text"].tolist()
embeddings = model.encode(texts, convert_to_numpy=True)

# Build the FAISS vector index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)


def retrieve_similar(query, k=7, emotion=None):

    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k * 3)

    results = df.iloc[indices[0]].copy()
    results["semantic_score"] = distances[0]

    # ---- Emotion weighted ranking ----
    if emotion:

        emotion = emotion.lower()

        results["emotion_bonus"] = results["emotion_normalized"].apply(
            lambda x: 0.15 if x == emotion else 0
        )

    else:
        results["emotion_bonus"] = 0

    # final score
    results["final_score"] = results["semantic_score"] + results["emotion_bonus"]

    results = results.sort_values("final_score")

    return results[
        ["emotion_normalized", "situation", "dialogue", "response", "final_score"]
    ].head(k)