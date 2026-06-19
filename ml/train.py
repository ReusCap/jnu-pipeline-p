# ml/train.py
import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from mlflow.tracking import MlflowClient

from app.config import (
    MLFLOW_TRACKING_URI,
    TRAIN_FILE_NAME,
    TEST_FILE_NAME,
    MODEL_NAME,
    ARTIFACT_DIR_NAME,
    DATA_DIR_NAME,
)
from ml.model_promoter import promote_if_better

BASE_DIR = os.path.dirname(__file__)
TRAIN_DATA_PATH = os.path.join(BASE_DIR, DATA_DIR_NAME, TRAIN_FILE_NAME)
TEST_DATA_PATH = os.path.join(BASE_DIR, DATA_DIR_NAME, TEST_FILE_NAME)
ARTIFACT_DIR = os.path.join(BASE_DIR, ARTIFACT_DIR_NAME)
MODEL_PATH = os.path.join(ARTIFACT_DIR, MODEL_NAME)

os.makedirs(ARTIFACT_DIR, exist_ok=True)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_registry_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("movie-sentiment-server")

train_df = pd.read_csv(TRAIN_DATA_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)
X_train, y_train = train_df["text"], train_df["label"]
X_test, y_test = test_df["text"], test_df["label"]

models = {
    "LogisticRegression": LogisticRegression(max_iter=200),
    "NaiveBayes": MultinomialNB(),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
}

client = MlflowClient()
best_test_acc = -1.0
best_version = None

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("vectorizer", "CountVectorizer")
        mlflow.log_param("train_row_count", len(train_df))
        mlflow.log_param("test_row_count", len(test_df))

        pipeline = Pipeline([
            ("vectorizer", CountVectorizer()),
            ("classifier", model),
        ])
        pipeline.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, pipeline.predict(X_train))
        test_acc = accuracy_score(y_test, pipeline.predict(X_test))
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)

        joblib.dump(pipeline, MODEL_PATH)  # 폴백용 로컬 파일(마지막 run 기준)
        mlflow.log_artifact(MODEL_PATH)
        mlflow.sklearn.log_model(
            pipeline, name="model", registered_model_name="movie-sentiment-model"
        )

        print(f"[{model_name}] train_acc={train_acc:.4f} test_acc={test_acc:.4f}")

        # 방금 등록된 최신 버전 추적
        latest_versions = client.search_model_versions("name='movie-sentiment-model'")
        latest_version = max(latest_versions, key=lambda v: int(v.version)).version
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_version = latest_version

# 이번 학습의 best가 기존 champion보다 좋으면 자동 승격
if best_version is not None:
    promote_if_better(best_version, best_test_acc)