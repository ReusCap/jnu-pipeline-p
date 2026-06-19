# app/retrain_issue.py
import logging
from datetime import datetime

from app.config import LOW_CONFIDENCE_LIMIT
from app.issue import create_github_issue

logger = logging.getLogger("sentiment")

# 서버 실행 동안만 유지되는 간단한 상태
_state = {
    "low_confidence_count": 0,
    "samples": [],
    "issue_created": False,
}


def update_issue_state(text: str, label: str, confidence: float, threshold: float):
    """낮은 confidence가 누적되면(drift 의심) GitHub Issue 생성."""
    if confidence < threshold:
        _state["low_confidence_count"] += 1
        _state["samples"].append({
            "text": text,
            "label": label,
            "confidence": round(float(confidence), 4),
            "time": datetime.now().isoformat(timespec="seconds"),
        })

        if (
            _state["low_confidence_count"] >= LOW_CONFIDENCE_LIMIT
            and not _state["issue_created"]
        ):
            create_drift_issue()
            _state["issue_created"] = True

    return _state


def create_drift_issue():
    samples = _state["samples"][-5:]
    title = "[MLOps] Drift suspected (low confidence accumulation)"
    body = (
        "## Drift Detection Report\n"
        "Low-confidence predictions accumulated.\n"
        f"- count: {_state['low_confidence_count']}\n"
        f"- threshold: {LOW_CONFIDENCE_LIMIT}\n\n"
        "## Recent Samples\n"
    )
    for s in samples:
        body += f"- ({s['confidence']}) {s['text']}\n"
    body += (
        "\n## Action\n"
        "- Please review data\n"
        "- Decide whether retraining is needed\n"
    )
    create_github_issue(title, body, logger)