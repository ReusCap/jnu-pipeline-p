# app/model.py

POSITIVE_WORDS = (
    "좋", "최고", "재밌", "감동", "추천",
    "훌륭", "완벽", "멋", "대박", "신기",
    "흥미", "몰입", "설레", "웃기", "유익",
    "아름", "따뜻", "행복", "즐거", "기대이상",
    "명작", "걸작", "인생작", "강추", "꿀잼",
    "소름", "뭉클", "짱", "굿", "갓",
)

NEGATIVE_WORDS = (
    "싫", "재미없", "별로", "최악", "지루", "노잼", "쓰레기",
    "실망", "졸", "억지", "뻔", "구리", "형편없", "후회",
    "낭비", "최하", "노재미", "망작", "혹평", "조악",
    "어색", "불편", "끔찍", "역겨", "최저", "돈아까",
    "시간낭비", "탈주", "하차", "폭망", "보지마",
)

NEGATION_PREFIX_TOKENS = frozenset({"안", "못", "별로", "전혀", "결코"})
NEGATION_SUFFIX_MARKERS = ("않", "없")


def _find_match(token, words):
    return next((w for w in words if w in token), None)


def _has_external_negation(tokens, idx):
    prev_token = tokens[idx - 1] if idx > 0 else ""
    next_token = tokens[idx + 1] if idx + 1 < len(tokens) else ""

    if prev_token in NEGATION_PREFIX_TOKENS or next_token in NEGATION_PREFIX_TOKENS:
        return True
    if any(m in next_token for m in NEGATION_SUFFIX_MARKERS):
        return True
    return False


def _has_internal_negation(token, matched_word):
    # 매칭된 감정 단어 자체가 포함하는 '없/않'은 self marker이므로 제외하고 검사
    remainder = token.replace(matched_word, "", 1)
    return any(m in remainder for m in NEGATION_SUFFIX_MARKERS)


def _is_negated(tokens, idx, matched_word):
    return (
        _has_external_negation(tokens, idx)
        or _has_internal_negation(tokens[idx], matched_word)
    )


def predict_sentiment(text: str):
    tokens = text.split()
    pos_score = 0
    neg_score = 0

    for i, tok in enumerate(tokens):
        pos_match = _find_match(tok, POSITIVE_WORDS)
        if pos_match:
            if _is_negated(tokens, i, pos_match):
                neg_score += 1
            else:
                pos_score += 1

        neg_match = _find_match(tok, NEGATIVE_WORDS)
        if neg_match:
            if _is_negated(tokens, i, neg_match):
                pos_score += 1
            else:
                neg_score += 1

    if pos_score > neg_score:
        return {"label": "긍정", "confidence": 0.7}
    if neg_score > pos_score:
        return {"label": "부정", "confidence": 0.7}
    return {"label": "중립", "confidence": 0.5}
