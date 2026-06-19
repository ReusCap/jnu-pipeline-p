# app/model_loader.py
import random
import joblib
import mlflow
import mlflow.sklearn
import mlflow.models
from mlflow.tracking import MlflowClient

from app.config import (
    MLFLOW_TRACKING_URI,
    MODEL_URI,
    LOCAL_MODEL_PATH,
    CHAMPION_MODEL_URI,
    CHALLENGER_MODEL_URI,
    CANARY_ENABLED,
    CANARY_RATIO,
)

# under bar로 시작하면 private이라는 관례적 표기
_model = None
_champion_model = None
_challenger_model = None


def load_model():
    """단일 모델 로드(레지스트리 champion → 실패 시 로컬 joblib 폴백)."""
    global _model
    if _model is not None:
        return _model
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_registry_uri(MLFLOW_TRACKING_URI)
        _model = mlflow.sklearn.load_model(MODEL_URI)
    except Exception:
        _model = joblib.load(LOCAL_MODEL_PATH)
    return _model


def load_champion_model():
    global _champion_model
    if _champion_model is None:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        _champion_model = mlflow.sklearn.load_model(CHAMPION_MODEL_URI)
    return _champion_model


def load_challenger_model():
    global _challenger_model
    if _challenger_model is None:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        _challenger_model = mlflow.sklearn.load_model(CHALLENGER_MODEL_URI)
    return _challenger_model


def select_serving_model():
    """카나리: 확률적으로 challenger, 아니면 champion.
    레지스트리 로드 실패 시 단일 모델(load_model)로 폴백."""
    try:
        if CANARY_ENABLED and random.random() < CANARY_RATIO:
            return load_challenger_model(), "challenger"
        return load_champion_model(), "champion"
    except Exception:
        return load_model(), "fallback"


def get_model_info(serving_model: str = "champion"):
    """serving_model(champion/challenger)에 해당하는 메타정보 반환.
    카나리는 요청마다 모델이 달라질 수 있어 캐시하지 않는다."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        if serving_model == "challenger":
            info = mlflow.models.get_model_info(CHALLENGER_MODEL_URI)
        else:
            info = mlflow.models.get_model_info(CHAMPION_MODEL_URI)
        run = MlflowClient().get_run(info.run_id)
        return {
            "run_id": info.run_id,
            "model_type": run.data.params.get("model_type"),
            "test_accuracy": run.data.metrics.get("test_accuracy"),
        }
    except Exception:
        return {
            "run_id": "unknown",
            "model_type": None,
            "test_accuracy": None,
        }