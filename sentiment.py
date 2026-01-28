import re
from pathlib import Path

import pandas as pd
from transformers import pipeline

FILE_PATH = "Cleaned Files/Press Conference/2025/12-10-25PC_cleaned.txt"


# keywords
CORE_POLICY_TERMS = [
    "inflation",
    "price stability",
    "federal funds rate",
    "fed funds rate",
    "interest rate",
    "policy stance",
    "monetary policy",
    "restrictive",
    "restriction",
    "easing",
    "tightening",
    "rate cut",
    "rate cuts",
    "rate hike",
    "rate hikes",
    "target range",
    "forward guidance",
    "financial conditions",
    "neutral rate",
    "terminal rate",
    "balance of risks",
    "policy restraint",
    "disinflation",
    "inflation expectations",
]

MACRO_TERMS = [
    "labor market",
    "employment",
    "unemployment",
    "job gains",
    "wages",
    "economic activity",
    "growth",
    "consumer spending",
    "investment",
    "credit conditions",
    "lending",
    "housing",
    "productivity",
    "downside risks",
    "upside risks",
    "uncertainty",
    "headwinds",
    "soft landing",
    "recession",
    "tariffs",
    "supply chains",
    "demand",
    "supply",
]


def normalize_whitespace(text: str) -> str:
    # Normalize whitespace across the text file
    return re.sub(r"\s+", " ", text).strip()


def sent_split(text: str):
    # Split sentences into list where there is punctuation follow by a space
    return re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)


def sentence_weight(sentence: str) -> float:
    # Used keywords to weight certain sentences more heavily
    # Sentences with core policy terms get weight of 2
    # Sentences with macroeconomic terms get weight of 1.5
    # All other sentences are weighted normally (1)
    s = sentence.lower()
    if any(term in s for term in CORE_POLICY_TERMS):
        return 2.0
    if any(term in s for term in MACRO_TERMS):
        return 1.5
    return 1.0


def analyze_meeting(file_path: str, device: int = 0, batch_size: int = 32, min_words: int = 6, save_sentence_csv: bool = False,):
    path = Path(file_path)

    m = re.match(r"(\d{2}-\d{2}-\d{2})", path.name)
    meeting_mmddyy = m.group(1) if m else None

    raw_text = path.read_text(encoding="utf-8", errors="ignore")
    text = normalize_whitespace(raw_text)

    sentences = sent_split(text)
    sentences = [s.strip() for s in sentences if s and len(s.split()) >= min_words]

    # Load the FinBERT model using HuggingFace transformers
    clf = pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        top_k=None,
        device=device,
        truncation=True,
    )

    outputs = clf(sentences, batch_size=batch_size)

    rows = []
    # Pair each sentence with its output
    for sent, out in zip(sentences, outputs):
        score_map = {d["label"].lower(): float(d["score"]) for d in out}
        pos = score_map.get("positive", 0.0)
        neg = score_map.get("negative", 0.0)
        neu = score_map.get("neutral", 0.0)

        tone = pos - neg               # tone scalar which ranges from +1 to -1
        w = sentence_weight(sent)

        rows.append(
            {
                "sentence": sent,
                "positive": pos,
                "negative": neg,
                "neutral": neu,
                "tone": tone,
                "weight": w,
                "weighted_tone": tone * w,
            }
        )

    df = pd.DataFrame(rows)

    # Weight meetings to more accurately reflect important content
    weights_sum = df["weight"].sum()
    meeting_weighted_tone = df["weighted_tone"].sum() / weights_sum
    meeting_unweighted_tone = df["tone"].mean()

    # Percentage of sentences with high negative / positive tails
    neg_tail_share = (df["negative"] > 0.60).mean()
    pos_tail_share = (df["positive"] > 0.60).mean()
    tone_std = df["tone"].std()

    # Caution ratio: high negative tail vs positive tail
    caution_ratio = neg_tail_share / (pos_tail_share + 1e-9)

    result = {
        "file": str(path),
        "meeting_mmddyy": meeting_mmddyy,
        "n_sentences": int(len(df)),
        "weighted_tone": float(meeting_weighted_tone),
        "unweighted_tone": float(meeting_unweighted_tone),
        "neg_tail_share": float(neg_tail_share),
        "pos_tail_share": float(pos_tail_share),
        "tone_std": float(tone_std),
        "caution_ratio": float(caution_ratio),
    }

    if save_sentence_csv:
        out_csv = path.with_suffix("").as_posix() + "_sentence_sentiment.csv"
        df.to_csv(out_csv, index=False)
        result["sentence_csv"] = out_csv

    return result


if __name__ == "__main__":
    summary = analyze_meeting(
        file_path=FILE_PATH,
        device=-1,
        batch_size=32,
        min_words=6,
        save_sentence_csv=True,
    )

    print("\n=== Meeting sentiment summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
