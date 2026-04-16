import gradio as gr

from emotion_detection import emotion_detector


def analyze_emotion(text):
    result = emotion_detector(text)

    if result["dominant_emotion"] is None:
        detail = result.get("_detail")
        if detail == "missing_token":
            msg = (
                "No Hugging Face token found. Add HF_TOKEN to your .env file "
                "(see README). The token must allow Inference API / Inference Providers."
            )
        elif detail == "inference_error":
            msg = (
                "Hugging Face inference failed (often 403: insufficient permissions). "
                "Create or edit your token at huggingface.co/settings/tokens and enable "
                "permission to call Inference Providers, then update HF_TOKEN in .env."
            )
        elif detail == "bad_response":
            msg = "The model returned an unexpected response. Try again in a moment."
        else:
            msg = "Invalid or empty text. Please enter a sentence with emotional context."
        return (
            msg,
            {
                "anger": 0.0,
                "disgust": 0.0,
                "fear": 0.0,
                "joy": 0.0,
                "sadness": 0.0,
            },
        )

    message = (
        f"Dominant emotion: {result['dominant_emotion']}\n\n"
        f"anger={result['anger']}, disgust={result['disgust']}"
        f"fear={result['fear']}, joy={result['joy']}, sadness={result['sadness']}"
    )

    chart_data = {
        "anger": float(result["anger"]),
        "disgust": float(result["disgust"]),
        "fear": float(result["fear"]),
        "joy": float(result["joy"]),
        "sadness": float(result["sadness"]),
    }
    return message, chart_data


demo = gr.Interface(
    fn=analyze_emotion,
    inputs=gr.Textbox(
        label="Text to analyze",
        lines=4,
        placeholder="Type a sentence, for example: I am very excited about this project.",
    ),
    outputs=[
        gr.Textbox(label="Result"),
        gr.Label(label="Emotion scores"),
    ],
    title="Emotion Detector",
    description="Free deployable emotion detector UI for Hugging Face Spaces.",
    flagging_mode="never",
)


if __name__ == "__main__":
    demo.launch(share=True)
