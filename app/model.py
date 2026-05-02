# app/model.py

_NEGATION_PREFIX_TOKENS = {"안", "못", "별로", "전혀", "결코"}
_NEGATION_SUFFIX_MARKERS = ("않", "없")


def _is_negated(tokens, idx):
    prev_token = tokens[idx - 1] if idx > 0 else ""
    next_token = tokens[idx + 1] if idx + 1 < len(tokens) else ""

    if prev_token in _NEGATION_PREFIX_TOKENS:
        return True
    if next_token in _NEGATION_PREFIX_TOKENS:
        return True
    if any(m in tokens[idx] for m in _NEGATION_SUFFIX_MARKERS):
        return True
    if any(m in next_token for m in _NEGATION_SUFFIX_MARKERS):
        return True
    return False


def predict_sentiment(text: str):
    positive_words = [
        "좋", "최고", "재밌", "감동", "추천",
        "훌륭", "완벽", "멋", "대박", "신기",
        "흥미", "몰입", "설레", "웃기", "유익",
        "아름", "따뜻", "행복", "즐거", "기대이상",
        "명작", "걸작", "인생작", "강추", "꿀잼",
        "소름", "뭉클", "짱", "굿", "갓"
    ]
    negative_words = [
        "싫", "재미없", "별로", "최악", "지루", "노잼", "쓰레기",
        "실망", "졸", "억지", "뻔", "구리", "형편없", "후회",
        "낭비", "최하", "노재미", "망작", "혹평", "조악",
        "어색", "불편", "끔찍", "역겨", "최저", "돈아까",
        "시간낭비", "탈주", "하차", "폭망", "보지마"
    ]

    tokens = text.split()
    pos_score = 0
    neg_score = 0

    for i, tok in enumerate(tokens):
        if any(w in tok for w in positive_words):
            if _is_negated(tokens, i):
                neg_score += 1
            else:
                pos_score += 1
        if any(w in tok for w in negative_words):
            if _is_negated(tokens, i):
                pos_score += 1
            else:
                neg_score += 1

    if pos_score > neg_score:
        return {"label": "긍정", "confidence": 0.7}
    elif neg_score > pos_score:
        return {"label": "부정", "confidence": 0.7}
    else:
        return {"label": "중립", "confidence": 0.5}
