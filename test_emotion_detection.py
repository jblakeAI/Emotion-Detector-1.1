import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import emotion_detection


class TestEmotionDetector(unittest.TestCase):
    def setUp(self):
        os.environ["HF_TOKEN"] = "hf_test_dummy"

    def tearDown(self):
        os.environ.pop("HF_TOKEN", None)
        os.environ.pop("HF_API_TOKEN", None)

    @patch.object(emotion_detection, "InferenceClient")
    def test_emotion_detector_response_shape_and_scores(self, mock_client_cls):
        def fake_classification(_text, **_kwargs):
            return [
                SimpleNamespace(label="anger", score=0.05),
                SimpleNamespace(label="disgust", score=0.05),
                SimpleNamespace(label="fear", score=0.05),
                SimpleNamespace(label="joy", score=0.75),
                SimpleNamespace(label="sadness", score=0.10),
            ]

        mock_client_cls.return_value.text_classification.side_effect = fake_classification

        texts = [
            "I am glad this happened.",
            "I am really mad about this.",
            "I feel disgusted just hearing about this.",
            "I am so sad about this.",
            "I am really afraid that this will happen.",
        ]
        emotion_keys = ["anger", "disgust", "fear", "joy", "sadness"]

        for text in texts:
            with self.subTest(text=text):
                response = emotion_detection.emotion_detector(text)

                for key in emotion_keys + ["dominant_emotion"]:
                    self.assertIn(key, response)

                scores = {}
                for key in emotion_keys:
                    self.assertIsInstance(response[key], (int, float))
                    self.assertGreaterEqual(response[key], 0.0)
                    self.assertLessEqual(response[key], 1.0)
                    scores[key] = float(response[key])

                expected_dominant = max(scores, key=scores.get)
                self.assertEqual(response["dominant_emotion"], expected_dominant)

        mock_client_cls.assert_called()

    def test_emotion_detector_blank_input(self):
        response = emotion_detection.emotion_detector("   ")

        self.assertEqual(
            response,
            {
                "anger": None,
                "disgust": None,
                "fear": None,
                "joy": None,
                "sadness": None,
                "dominant_emotion": None,
            },
        )

    def test_emotion_detector_no_token(self):
        del os.environ["HF_TOKEN"]
        response = emotion_detection.emotion_detector("Hello world")
        self.assertTrue(
            all(response[k] is None for k in ["anger", "disgust", "fear", "joy", "sadness"])
        )
        self.assertIsNone(response["dominant_emotion"])
        self.assertEqual(response.get("_detail"), "missing_token")


if __name__ == "__main__":
    unittest.main()
