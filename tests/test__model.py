# tests/test_model.py

from app.model import predict_sentiment

def test_predict_positive():
    result = predict_sentiment("이 영화 정말 최고! 강력 추천합니다.")
    assert result["label"] == "긍정"
    assert result["confidence"] == 0.7

def test_predict_negative():
    result = predict_sentiment("너무 지루하고 노잼 쓰레기 영화")
    assert result["label"] == "부정"
    assert result["confidence"] == 0.7

def test_predict_neutral():
    result = predict_sentiment("그냥 평범한 영화네요.")
    assert result["label"] == "중립"