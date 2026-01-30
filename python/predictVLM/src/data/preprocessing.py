from predictVLM.src.data.utils import get_valid_date_range, mmr
from predictVLM.src.data.vector_db.query_db import search_news_by_embedding

import os
import ast
import numpy as np
import pandas as pd
import json
import boto3
from tqdm import tqdm
from copy import deepcopy


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


def add_price(data, price_df, window_size, spans):
    price_df = price_df.copy()
    
    for span in spans:
        price_df[f'EMA_{span}'] = price_df['close'].ewm(span=span, adjust=False).mean()
    
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

        data[i]["time"] = window['open'].tolist()
        data[i]["open"] = window['open'].tolist()
        data[i]["high"] = window['high'].tolist()
        data[i]["low"] = window['low'].tolist()
        data[i]['close'] = window['close'].tolist()
        data[i]['volume'] = window['Volume'].tolist()
        
        for span in spans:
            data[i][f'EMA_{span}'] = window[f'EMA_{span}'].tolist()
        
        
def add_initial_prompt(data, name, window_size, horizons, spans):
    selected_columns = ["time", "open", "high", "low", "close"]

    for span in spans:
        selected_columns.append(f'EMA_{span}')
        
    selected_columns.append("volume")
    
    for i in range(len(data)):
        data[i]['initial_prompt'] = f"""Given the past {window_size} timesteps of {name} futures data (columns: {', '.join(selected_columns)}),
generate a concise query to retrieve relevant news articles that may impact future closing prices.

[Objective]
The retrieved news will be used as retrieval-augmented context for predicting {name} futures closing prices at
{', '.join(["t" if (h-1) == 0 else f"t+{h-1}" for h in horizons])}. Focus on information affecting short- to mid-term price movements.

[Time Series Values]
{", ".join(f"{col}={data[i][col]}" for col in selected_columns)}

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
        

def add_final_prompt(data, name, window_size, horizons, spans):
    selected_columns = ["time", "open", "high", "low", "close"]

    for span in spans:
        selected_columns.append(f'EMA_{span}')
        
    selected_columns.append("volume")
    
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
{", ".join(f"{col}={data[i][col]}" for col in selected_columns)}

[Candlestick Chart]
An image representing the same time window (t-{window_size} to t-1).

[Related News Articles]
{article}
[Output Format]
Return a Python dictionary with exactly the following keys and float values:
{
  {", ".join(f"{horizon_key(h)}: float" for h in horizons)}
}

Output only the dictionary. No additional text."""


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
        
        data[i]['output'] = str(output_dict)
        new_data.append(data[i])
    
    return new_data


def add_prev_close(data, date_list, price_df, horizons):
    offsets = [h-2 for h in horizons]
    
    for i in range(len(data)):
        t_index = date_list.index(data[i]['date'])
        prev_closes = []
        
        for offset in offsets:
            idx = t_index + offset

            if idx < 0 or idx >= len(date_list):
                prev_closes.append(None)
                continue

            prev_date = date_list[idx]
            row = price_df.loc[price_df['time'] == prev_date, 'close']

            if not row.empty:
                prev_closes.append(float(row.values[0]))
            else:
                prev_closes.append(None)
                
        data[i]['prev_close'] = prev_closes


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
        ],
        "prev_close": sample["prev_close"]
    }


def train_valid_test_split(data, split_file, index, data_dir, image_dir):    
    train_dates = split_file['folds'][index]['train']['t_dates']
    valid_dates =  split_file['folds'][index]['val']['t_dates']
    test_dates = split_file['meta']['fixed_test']['t_dates']
    
    train = []
    valid = []
    test = []

    for i in range(len(data)):
        if data[i]['date'] in train_dates:
            train.append({'image': os.path.join(image_dir, f"{data[i]['date']}.png"),
                          'prompt': data[i]['final_prompt'],
                          'output': data[i]['output'],
                          'prev_close': data[i]['prev_close']})
        elif data[i]['date'] in valid_dates:
            valid.append({'image': os.path.join(image_dir, f"{data[i]['date']}.png"),
                          'prompt': data[i]['final_prompt'],
                          'output': data[i]['output'],
                          'prev_close': data[i]['prev_close']})
        elif data[i]['date'] in test_dates:
            test.append({'image': os.path.join(image_dir, f"{data[i]['date']}.png"),
                         'prompt': data[i]['final_prompt'],
                         'output': data[i]['output'],
                         'prev_close': data[i]['prev_close']})

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
            

def preprocess_messages(messages, processor, include_assistant=True, add_eos=False):
    msgs = deepcopy(messages)
    
    if not include_assistant:
        msgs = [m for m in msgs if m["role"] != "assistant"]

    processed = []
    for msg in msgs:
        safe_content = []
        for c in msg["content"]:
            if c.get("type") == "image" and c.get("image"):
                safe_content.append({"type": "image", "image": c["image"]})
            elif c.get("type") == "text" and c.get("text"):
                safe_content.append({"type": "text", "text": c["text"]})

        msg["content"] = safe_content

        if include_assistant and msg["role"] == "assistant" and add_eos:
            text_idxs = [i for i, c in enumerate(msg["content"]) if c["type"] == "text"]
            if text_idxs:
                msg["content"][text_idxs[-1]]["text"] = "<|assistant|> " + msg["content"][text_idxs[-1]]["text"] + processor.tokenizer.eos_token

        processed.append(msg)

    return processed