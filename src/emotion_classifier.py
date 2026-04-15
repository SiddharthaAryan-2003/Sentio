from transformers import pipeline

emotion_model = pipeline(
    "text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=3
)

GOEMOTION_TO_CORE = {

    # JOY FAMILY
    "joy": "joy",
    "amusement": "joy",
    "excitement": "joy",
    "optimism": "hope",

    # SADNESS FAMILY
    "sadness": "sadness",
    "grief": "sadness",
    "disappointment": "sadness",

    # ANGER FAMILY
    "anger": "anger",
    "annoyance": "anger",

    # FEAR / ANXIETY
    "fear": "fear",
    "nervousness": "fear",

    # LOVE / ATTACHMENT
    "love": "love",
    "admiration": "love",
    "caring": "love",
    "gratitude": "love",

    # PRIDE
    "pride": "pride",

    # GUILT / SHAME
    "guilt": "guilt",
    "remorse": "guilt",

    # SURPRISE
    "surprise": "surprise",

    # DISGUST
    "disgust": "disgust"
}


def detect_emotion(text):

    predictions = emotion_model(text)[0]

    for pred in predictions:

        label = pred["label"]
        score = pred["score"]

        if label in GOEMOTION_TO_CORE:

            return GOEMOTION_TO_CORE[label], score

    return "neutral", 0.0