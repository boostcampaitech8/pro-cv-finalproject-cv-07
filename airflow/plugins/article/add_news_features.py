import gc

import torch
import pandas as pd
from tqdm import tqdm
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    pipeline,
)


def _build_text_input(df: pd.DataFrame) -> list[str]:
    return (df['title'].fillna('') + "\n\n" + df['description'].fillna('')).tolist()


def add_news_features(df: pd.DataFrame, batch_size: int = 256) -> pd.DataFrame:
    """sentiment + timeframe 을 한 번에 추가하고 모델 메모리를 해제한다."""
    df = df.copy()
    texts = _build_text_input(df)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Sentiment ──
    print("  [Feature] Sentiment analysis ...")
    sent_model = BertForSequenceClassification.from_pretrained(
        "ahmedrachid/FinancialBERT-Sentiment-Analysis", num_labels=3
    )
    sent_tokenizer = BertTokenizer.from_pretrained(
        "ahmedrachid/FinancialBERT-Sentiment-Analysis"
    )
    sent_pipe = pipeline("sentiment-analysis", model=sent_model, tokenizer=sent_tokenizer, device=device)

    sentiment_labels, sentiment_scores = [], []
    for i in tqdm(range(0, len(texts), batch_size), desc="Sentiment"):
        for r in sent_pipe(texts[i : i + batch_size], truncation=True, max_length=512):
            sentiment_labels.append(r['label'])
            sentiment_scores.append(r['score'])

    df['sentiment_label'] = [l.lower() for l in sentiment_labels]
    df['sentiment_score'] = sentiment_scores

    # 메모리 해제
    del sent_pipe, sent_model, sent_tokenizer
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # ── Timeframe ──
    print("  [Feature] Timeframe classification ...")
    tf_pipe = pipeline(
        "text-classification",
        model="ProfessorLeVesseur/bert-base-cased-timeframe-classifier",
        device=device,
    )

    timeframe_labels, timeframe_scores = [], []
    for i in tqdm(range(0, len(texts), batch_size), desc="Timeframe"):
        for r in tf_pipe(texts[i : i + batch_size], truncation=True, max_length=512):
            timeframe_labels.append(r['label'])
            timeframe_scores.append(r['score'])

    df['timeframe_label'] = [l.lower() for l in timeframe_labels]
    df['timeframe_score'] = timeframe_scores

    # 메모리 해제
    del tf_pipe
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return df
