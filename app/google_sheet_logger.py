# app/google_sheet_logger.py
import os
import json
from datetime import datetime

import gspread
from google.oauth2.service_account import Credentials

SCOPE = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

_spreadsheet = None


def _build_credentials():
    """Render는 JSON 문자열(GOOGLE_SERVICE_ACCOUNT_JSON),
    로컬은 키 파일(GOOGLE_SERVICE_ACCOUNT_FILE) — 둘 다 지원."""
    sa_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    sa_file = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
    if sa_json:
        return Credentials.from_service_account_info(json.loads(sa_json), scopes=SCOPE)
    if sa_file:
        return Credentials.from_service_account_file(sa_file, scopes=SCOPE)
    raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON / GOOGLE_SERVICE_ACCOUNT_FILE not set")


def get_spreadsheet():
    global _spreadsheet
    if _spreadsheet is not None:
        return _spreadsheet

    sheet_name = os.getenv("GOOGLE_SHEET_NAME")
    if not sheet_name:
        raise RuntimeError("GOOGLE_SHEET_NAME is not set")

    client = gspread.authorize(_build_credentials())
    _spreadsheet = client.open(sheet_name)
    return _spreadsheet


def append_prediction_log(text: str, label: str, confidence: float, serving_model: str):
    worksheet = get_spreadsheet().worksheet("prediction_logs")
    worksheet.append_row([
        datetime.now().isoformat(timespec="seconds"),
        text,
        label,
        round(float(confidence), 4),
        serving_model,
    ])


def append_feedback_log(text: str, prediction: str, correct_label: str, confidence: float, serving_model: str):
    worksheet = get_spreadsheet().worksheet("feedback_logs")
    worksheet.append_row([
        datetime.now().isoformat(timespec="seconds"),
        text,
        prediction,
        correct_label,
        round(float(confidence), 4),
        serving_model,
    ])