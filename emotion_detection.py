import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load .env from this project folder (works even if the shell cwd is elsewhere).
load_dotenv(Path(__file__).resolve().parent / ".env")


def _label_and_score(item):
    if isinstance(item, dict):
        return str(item.get("label", "")), float(item.get("score", 0.0))
    label = getattr(item, "label", "") or ""
    score = getattr(item, "score", 0.0)
    return str(label), float(score)


def _failed_response(reason):
    """Same shape as a failed run; optional _detail explains why (for UI messages)."""
    return {
        "anger": None,
        "disgust": None,
        "fear": None,
        "joy": None,
        "sadness": None,
        "dominant_emotion": None,
        "_detail": reason,
    }


def emotion_detector(text_to_analyze):
    """Call Hugging Face Inference (router) and return emotion scores."""
    blank = {
        "anger": None,
        "disgust": None,
        "fear": None,
        "joy": None,
        "sadness": None,
        "dominant_emotion": None,
    }

    if not text_to_analyze or not text_to_analyze.strip():
        return blank

    token = (os.getenv("HF_API_TOKEN") or os.getenv("HF_TOKEN") or "").strip()
    if not token:
        return _failed_response("missing_token")

    try:
        client = InferenceClient(token=token)
        predictions = client.text_classification(
            text_to_analyze.strip(),
            model="j-hartmann/emotion-english-distilroberta-base",
        )
    except Exception:
        return _failed_response("inference_error")

    if not predictions:
        return _failed_response("bad_response")

    allowed_labels = {"anger", "disgust", "fear", "joy", "sadness"}
    emotions = {key: 0.0 for key in allowed_labels}

    try:
        for item in predictions:
            label, score = _label_and_score(item)
            label = label.lower()
            if label in emotions:
                emotions[label] = score
    except (TypeError, ValueError):
        return _failed_response("bad_response")

    dominant_emotion = max(emotions, key=emotions.get)
    emotions["dominant_emotion"] = dominant_emotion
    return emotions
