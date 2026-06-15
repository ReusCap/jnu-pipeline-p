# app/config.py
import os

# 추론 모드: "rules"(규칙 기반) 또는 "ml"(학습 모델)
MODEL_MODE = os.getenv("MODEL_MODE", "rules")
LOCAL_MODEL_PATH = "ml/artifacts/sentiment_model.joblib"