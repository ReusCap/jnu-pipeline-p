# app/model_loader.py
import joblib
import mlflow
import mlflow.sklearn
import mlflow.models
from mlflow.tracking import MlflowClient

from app.config import MLFLOW_TRACKING_URI, MODEL_URI, LOCAL_MODEL_PATH

_model = None
_model_info = None


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


def get_model_info():
    """현재 서빙 중인 모델(@champion)의 run_id/model_type/test_accuracy를 반환.
    레지스트리 접근 실패 시 unknown 값으로 폴백한다."""
    global _model_info
    if _model_info is not None:
        return _model_info

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        info = mlflow.models.get_model_info(MODEL_URI)
        run = MlflowClient().get_run(info.run_id)
        _model_info = {
            "run_id": info.run_id,
            "model_type": run.data.params.get("model_type"),
            "test_accuracy": run.data.metrics.get("test_accuracy"),
        }
    except Exception:
        _model_info = {
            "run_id": "unknown",
            "model_type": None,
            "test_accuracy": None,
        }

    return _model_info