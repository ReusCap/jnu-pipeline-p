# tests/test_model.py
import os
import joblib


def test_trained_model_exists():
    assert os.path.exists("ml/artifacts/sentiment_model.joblib")


def test_model_can_predict():
    model = joblib.load("ml/artifacts/sentiment_model.joblib")
    pred1 = model.predict(["정말 재밌고 감동적인 영화"])[0]
    pred2 = model.predict(["지루하고 재미없는 영화"])[0]
    assert pred1 in ["긍정", "부정"]
    assert pred2 in ["긍정", "부정"]