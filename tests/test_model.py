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

def test_predict_negation_prefix():
    result = predict_sentiment("이 영화 정말 안 좋다")
    assert result["label"] == "부정"

def test_predict_negation_suffix():
    result = predict_sentiment("재밌지 않다")
    assert result["label"] == "부정"

def test_predict_negation_postfix_token():
    result = predict_sentiment("이거 추천 안 합니다")
    assert result["label"] == "부정"

def test_predict_combined_negation():
    result = predict_sentiment("별로 좋지 않다")
    assert result["label"] == "부정"