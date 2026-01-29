from predictVLM.src.data.utils import get_valid_date_range, mmr
from predictVLM.src.data.vector_db.query_db import search_news_by_embedding

import os
import ast
import numpy as np
import pandas as pd
import json
import boto3
from tqdm import tqdm


def filtering_price(df):
    df = df.copy()
    
    df = df[(df['time'] >= "2017-11-09") & (df['time'] <= "2025-11-14")]
    
    return df


def filtering_news(df):
    df = df.copy()
    
    df = df[df["filter_status"] == 'T']
    
    return df


def str_to_python(obj_str):
    if obj_str is None:
        return None
    return ast.literal_eval(str(obj_str).strip())


def handle_holidays(date_df, df):
    date_df = date_df.copy()
    df = df.copy()
    
    date_df['time'] = pd.to_datetime(date_df['time']).dt.date
    df['publish_date'] = pd.to_datetime(df['publish_date']).dt.date

    trade_days = date_df['time'].sort_values().unique()
    df['trade_date'] = df['publish_date'].apply(
        lambda d: trade_days[trade_days <= d][-1] if np.any(trade_days <= d) else pd.NaT
    )
    df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
    
    return df


def add_price(data, price_df, window_size):
    for i in range(len(data)):
        t_idx = price_df[price_df['time'] == data[i]['date']].index
    
        if len(t_idx) == 0:
            continue
        t_idx = t_idx[0]
    
        start_idx = t_idx - window_size
        end_idx = t_idx - 1
    
        if start_idx < 0:
            continue

        window = price_df.iloc[start_idx:end_idx+1]
    
        data[i]["opens"] = window['open'].tolist()
        data[i]["highs"] = window['high'].tolist()
        data[i]["lows"] = window['low'].tolist()
        data[i]['closes'] = window['close'].tolist()
        data[i]['EMAs'] = window['EMA'].tolist()
        data[i]['Volumes'] = window['Volume'].tolist()
        
        
def add_initial_prompt(data, name, window_size, horizons):
    for i in range(len(data)):
        data[i]['initial_prompt'] = f"""Given the past {window_size} timesteps of {name} futures data (columns: open, high, low, close, EMA, Volume),
generate a concise query to retrieve relevant news articles that may impact future closing prices.

[Objective]
The retrieved news will be used as retrieval-augmented context for predicting {name} futures closing prices at
{', '.join(["t" if (h-1) == 0 else f"t+{h-1}" for h in horizons])}. Focus on information affecting short- to mid-term price movements.

[Time Series Values]
open={data[i]["opens"]}, high={data[i]["highs"]}, low={data[i]["lows"]}, close={data[i]["closes"]}, EMA={data[i]["EMAs"]}, Volume={data[i]["Volumes"]}

[Instruction]
Create a short query for news retrieval related to {name} futures.
Emphasize supply-demand dynamics, weather, macroeconomic factors, policy, and global trade."""


def add_embedding(data, access_key_id, secret_access_key):
    class TitanEmbeddings(object):
        accept = "application/json"
        content_type = "application/json"
    
        def __init__(self, model_id="amazon.titan-embed-text-v2:0"):
            self.bedrock = boto3.client(service_name='bedrock-runtime',
                                        region_name='us-east-1',
                                        aws_access_key_id=access_key_id,
                                        aws_secret_access_key=secret_access_key)
            self.model_id = model_id
        def __call__(self, text, dimensions, normalize=True):
            body = json.dumps({
                "inputText": text,
                "dimensions": dimensions,
                "normalize": normalize
            })
            response = self.bedrock.invoke_model(
                body=body, modelId=self.model_id, accept=self.accept, contentType=self.content_type
            )
            response_body = json.loads(response.get('body').read())
            return response_body['embedding']
    
    titan_embeddings_v2 = TitanEmbeddings(model_id="amazon.titan-embed-text-v2:0")
    
    for i in tqdm(range(len(data)), desc="Embedding generation"):
        input_text = data[i]['initial_prompt']
        embedding = titan_embeddings_v2(input_text, dimensions=512, normalize=True)
        data[i]['embedding'] = embedding
        
        
def add_relative_news(data, date_list, host, port, top_k=5):
    for i in tqdm(range(len(data)), desc="Search News"):
        valid_dates = get_valid_date_range(data[i]["date"], date_list)
        
        response = search_news_by_embedding(host, port, "news", data[i]["embedding"], valid_dates)
        results = response["result"]
    
        doc_embeddings = [r["vector"] for r in results]
        docs = [r["payload"] for r in results]
    
        selected_idx = mmr(
            query_emb=data[i]["embedding"],
            doc_embs=doc_embeddings,
            lambda_=0.7,
            top_k=top_k
        )
    
        final_docs = [docs[i] for i in selected_idx]
    
        titles = []
        descriptions = []
    
        for doc in final_docs:
            titles.append(doc['title'])
            descriptions.append(doc['description'])
    
        data[i]["title"] = titles
        data[i]["description"] = descriptions
        

def add_final_prompt(data, name, window_size, horizons):
    def horizon_key(h):
        h = h - 1
        return "t" if h == 0 else f"t+{h}"
    
    output_fields = ",\n  ".join(
        f"\"{horizon_key(h)}\": float" for h in horizons
    )
          
    for i in range(len(data)):
        article = ""
        if len(data[i]["title"]) == 0:
            article = "None\n"
        else:
            for j in range(len(data[i]["title"])):
                article += f"{j+1}. Title: '{data[i]['title'][j]}'\n   Description: '{data[i]['description'][j]}'\n"
    
        data[i]['final_prompt'] = f"""You are a multimodal forecasting model for financial time series.

Your task is to predict future {name} closing prices by jointly considering:
- recent numerical time-series patterns,
- visual trends in the candlestick chart,
- and relevant external information from news articles.

[Prediction Targets]
Predict the closing prices at {", ".join(horizon_key(h) for h in horizons)}.

[Time Series Values] (t-{window_size} to t-1)
open={data[i]["opens"]}, high={data[i]["highs"]}, low={data[i]["lows"]}, close={data[i]["closes"]}, EMA={data[i]["EMAs"]}, Volume={data[i]["Volumes"]}

[Candlestick Chart]
An image representing the same time window (t-{window_size} to t-1).

[Related News Articles]
{article}
[Output Format]
Return a JSON object with exactly four fields:
{{
  {output_fields}
}}

Output only valid JSON. No additional text."""


def add_output(data, date_list, price_df, horizons):
    new_data = []
    horizons = [h-1 for h in horizons]
    
    for i in range(len(data)):
        t_index = date_list.index(data[i]['date'])

        target_dates = []
        valid_horizons = []

        skip = False

        for h in horizons:
            if t_index + h < len(date_list):
                target_dates.append(date_list[t_index + h])
                valid_horizons.append(h)
            else:
                print(f"Skipping horizon t+{h} for date {data[i]['date']} (out of range)")
                skip = True
                break 
        
        if skip:
            continue

        close_values = price_df.loc[price_df['time'].isin(target_dates), 'close'].values

        output_dict = {}
        for h, v in zip(valid_horizons, close_values):
            if h == 0:
                output_dict["t"] = float(v)
            else:
                output_dict[f"t+{h}"] = float(v)
        
        data[i]['output'] = json.dumps(output_dict)
        new_data.append(data[i])
    
    return new_data


def to_chatml(sample):
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": sample["image"]
                    },
                    {
                        "type": "text",
                        "text": sample["prompt"]
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": sample["output"]
                    }
                ]
            }
        ]
    }


# TODO : 이미지 완성되면 여기 절대 경로 수정해야 함!!
def train_valid_test_split(data, split_file, index, data_dir, name):    
    train_dates = split_file['folds'][index]['train']['t_dates']
    valid_dates =  split_file['folds'][index]['val']['t_dates']
    test_dates = split_file['meta']['fixed_test']['t_dates']
    
    train = []
    valid = []
    test = []

    for i in range(len(data)):
        if data[i]['date'] in train_dates:
            train.append({'image': f'/data/ephemeral/home/outputs/{name}/{data[i]["date"]}.png',
                          'prompt': data[i]['final_prompt'],
                          'output': data[i]['output']})
        elif data[i]['date'] in valid_dates:
            valid.append({'image': f'/data/ephemeral/home/outputs/{name}/{data[i]["date"]}.png',
                          'prompt': data[i]['final_prompt'],
                          'output': data[i]['output']})
        elif data[i]['date'] in test_dates:
            test.append({'image': f'/data/ephemeral/home/outputs/{name}/{data[i]["date"]}.png',
                         'prompt': data[i]['final_prompt'],
                         'output': data[i]['output']})

    with open(os.path.join(data_dir, "preprocessing/train.jsonl"), "w", encoding="utf-8") as f:
        for sample in train:
            chatml = to_chatml(sample)
            f.write(json.dumps(chatml, ensure_ascii=False) + "\n")

    with open(os.path.join(data_dir, "preprocessing/val.jsonl"), "w", encoding="utf-8") as f:
        for sample in valid:
            chatml = to_chatml(sample)
            f.write(json.dumps(chatml, ensure_ascii=False) + "\n")

    with open(os.path.join(data_dir, "preprocessing/test.jsonl"), "w", encoding="utf-8") as f:
        for sample in test:
            chatml = to_chatml(sample)
            f.write(json.dumps(chatml, ensure_ascii=False) + "\n")