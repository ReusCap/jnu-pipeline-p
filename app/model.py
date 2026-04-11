# app/model.py

def predict_sentiment(text: str):
    positive_words = ["좋", "최고", "재밌", "감동", "추천"]
    negative_words = ["별로", "최악", "지루", "노잼", "쓰레기"]

    pos_score = sum(1 for w in positive_words if w in text)
    neg_score = sum(1 for w in negative_words if w in text)

    if pos_score > neg_score:
        return {"label": "긍정", "confidence": 0.7}
    elif neg_score > pos_score:
        return {"label": "부정", "confidence": 0.7}
    else:
        return {"label": "중립", "confidence": 0.5}