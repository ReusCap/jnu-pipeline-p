# app/model_loader.py
import joblib
import mlflow
import mlflow.sklearn

from app.config import MLFLOW_TRACKING_URI, MODEL_URI, LOCAL_MODEL_PATH

_model = None


def load_model():
    """MLflow 레지스트리(champion)에서 우선 로드하고,
    접근 실패 시 로컬 joblib 아티팩트로 폴백한다."""
    global _model
    if _model is not None:
        return _model

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_registry_uri(MLFLOW_TRACKING_URI)
        _model = mlflow.sklearn.load_model(MODEL_URI)
    except Exception:
        # 레지스트리 접근 불가(별칭 미설정/서버 다운) 시 로컬 모델 사용
        _model = joblib.load(LOCAL_MODEL_PATH)

    return _model