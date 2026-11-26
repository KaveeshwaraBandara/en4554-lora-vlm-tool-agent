"""VQA accuracy (VQAv2 / TextVQA / OK-VQA) and ANLS (DocVQA).

VQA accuracy follows the official VQA evaluation script (vqaEval.py, Antol et al.):
answers are normalised (punctuation, number words -> digits, articles, a subset of the
contraction map), and the score for a prediction is the average over all
leave-one-annotator-out subsets of min(1, #matching answers / 3).

ANLS follows the DocVQA definition (Biten et al., 2019): for each ground truth,
1 - normalised Levenshtein distance if that distance is < 0.5, else 0; take the max
over ground truths. Comparison is case-insensitive and whitespace-trimmed.
"""

from __future__ import annotations

import re

# --- VQA answer normalisation (official vqaEval.py) -------------------------------

_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't",
    "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", "hasnt": "hasn't",
    "havent": "haven't", "hes": "he's", "im": "i'm", "ive": "i've", "isnt": "isn't", "itd": "it'd",
    "itll": "it'll", "lets": "let's", "shouldnt": "shouldn't", "thats": "that's", "theres": "there's",
    "theyre": "they're", "theyve": "they've", "wasnt": "wasn't", "werent": "weren't", "whats": "what's",
    "wheres": "where's", "whos": "who's", "wont": "won't", "wouldnt": "wouldn't", "youre": "you're",
    "youve": "you've",
}
_NUMBER_WORDS = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
_ARTICLES = {"a", "an", "the"}
_PUNCT = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-",
          ">", "<", "@", "`", ",", "?", "!"]
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(,)(\d)")


def _process_punctuation(text: str) -> str:
    out = text
    for p in _PUNCT:
        if (p + " " in text or " " + p in text) or (_COMMA_STRIP.search(text) is not None):
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    return _PERIOD_STRIP.sub("", out)


def _process_digit_article(text: str) -> str:
    words = []
    for w in text.lower().split():
        w = _NUMBER_WORDS.get(w, w)
        if w in _ARTICLES:
            continue
        words.append(_CONTRACTIONS.get(w, w))
    return " ".join(words)


def normalize_vqa(ans: str) -> str:
    ans = ans.replace("\n", " ").replace("\t", " ").strip()
    return _process_digit_article(_process_punctuation(ans))


def vqa_accuracy(pred: str, answers: list[str]) -> float:
    """Official VQA accuracy for one prediction against the annotator answers."""
    if not answers:
        return 0.0
    p = normalize_vqa(pred)
    gts = [normalize_vqa(a) for a in answers]
    if len(gts) == 1:
        return float(p == gts[0])
    accs = []
    for i in range(len(gts)):
        others = gts[:i] + gts[i + 1:]
        accs.append(min(1.0, sum(g == p for g in others) / 3.0))
    return sum(accs) / len(accs)


# --- ANLS --------------------------------------------------------------------------

def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def anls(pred: str, answers: list[str], threshold: float = 0.5) -> float:
    p = " ".join(pred.strip().lower().split())
    best = 0.0
    for gt in answers:
        g = " ".join(gt.strip().lower().split())
        if not p and not g:
            return 1.0
        nl = _levenshtein(p, g) / max(len(p), len(g))
        best = max(best, 1.0 - nl if nl < threshold else 0.0)
    return best


METRICS = {"vqa_acc": vqa_accuracy, "anls": anls}


def score(metric: str, pred: str, answers: list[str]) -> float:
    return METRICS[metric](pred, answers)
