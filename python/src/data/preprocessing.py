import pandas as pd
from tqdm import tqdm
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
from sklearn.preprocessing import MaxAbsScaler


def filtering_news(df):
    df = df.copy()
    
    df = df[df["filter_status"] == 'T']
    
    return df


def add_news_sentiment(df, batch_size=256):
    df = df.copy()
    
    # 모델 정의
    model = BertForSequenceClassification.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis",num_labels=3)
    tokenizer = BertTokenizer.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)

    # input 정의
    df['text_input'] = df['title'].fillna('') + "\n\n" + df['description'].fillna('')
    
    # 긍부정 tagging
    sentiment_labels = []
    sentiment_scores = []

    for i in tqdm(range(0, len(df), batch_size)):
        batch_texts = df['text_input'].iloc[i:i+batch_size].tolist()
        results = nlp(batch_texts)
        for r in results:
            sentiment_labels.append(r['label'])
            sentiment_scores.append(r['score'])

    df['sentiment_label'] = sentiment_labels
    df['sentiment_score'] = sentiment_scores
    
    df.drop(columns=['text_input'], inplace=True)
    return df
    

def add_news_timeframe(df, batch_size=256):
    df = df.copy()
    
    # 모델 정의
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = pipeline(
        "text-classification",
        model="ProfessorLeVesseur/bert-base-cased-timeframe-classifier",
        device=device
    )

    # input 정의
    df['text_input'] = df['title'].fillna('') + "\n\n" + df['description'].fillna('')

    # 시제 tagging
    timeframe_labels = []
    timeframe_scores = []

    for i in tqdm(range(0, len(df), batch_size)):
        batch_texts = df['text_input'].iloc[i:i+batch_size].tolist()
        results = classifier(batch_texts)
        for r in results:
            timeframe_labels.append(r['label'])
            timeframe_scores.append(r['score'])

    df['timeframe_label'] = timeframe_labels
    df['timeframe_score'] = timeframe_scores

    df.drop(columns=['text_input'], inplace=True)
    return df


