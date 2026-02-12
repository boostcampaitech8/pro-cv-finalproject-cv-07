import os
import argparse
from dotenv import load_dotenv

import pandas as pd
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny

import requests
from sklearn.metrics.pairwise import cosine_similarity

import mysql.connector
from datetime import datetime


TYPE_TO_TYPES = {
    "corn": ["corn", "crop-general"],
    "wheat": ["wheat", "crop-general"],
    "soybean": ["soybean", "crop-general"],
    "gold": ["gold"],
    "sliver": ["sliver"],
    "copper": ["copper"]
}

TYPE_TO_TICKER = {
    "corn": "ZC=F",
    "wheat": "ZW=F",
    "soybean": "ZS=F",
    "gold": "GC=F",
    "sliver": "SI=F",
    "copper": "HG=F",
    "crop-general": None
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-date",
        type=str,
        default="2025-10-30"
    )
    parser.add_argument(
        "--type",
        type=str,
        default="corn",
    )
    
    return parser.parse_args()


def connect_vector_db(host, port):
    client = QdrantClient(host=host, port=port)
    return client


def fetch_embeddings_by_date(client, collection_name, date_str, types):
    points, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="collect_date",
                    match=MatchValue(value=date_str)
                ),
                FieldCondition(
                    key="type",
                    match=MatchAny(any=types)
                )
            ]
        ),
        with_vectors=True,
        limit=10_000 
    )

    return [p.vector for p in points]


def mean_pool(vectors):
    if len(vectors) == 0:
        return None
    pooled = np.mean(np.stack(vectors), axis=0)
    return pooled.tolist()


def search_news_by_embedding(host, port, collection_name, query_embedding, valid_dates, top_k=100):
    url = f"http://{host}:{port}/collections/{collection_name}/points/search"

    payload = {
        "vector": query_embedding,
        "limit": top_k,
        "with_payload": True, 
        "with_vector": True,
        "filter": {
            "must": [
                {
                    "key": "collect_date",
                    "match": {"any": valid_dates}
                }
            ]
        }
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()


def mmr_weighted(query_emb, doc_embs, doc_dates, temporal_dict, lambda_=0.8, top_k=5):
    selected = []
    candidates = list(range(len(doc_embs)))

    query_emb = np.array(query_emb).reshape(1, -1)
    doc_embs = np.array(doc_embs)

    while len(selected) < top_k and candidates:
        scores = []
        max_weight = max(temporal_dict.values())
        
        for i in candidates:
            raw_w = temporal_dict.get(doc_dates[i], 1.0)
            weight = raw_w / max_weight
            
            sim_to_query = cosine_similarity(query_emb, doc_embs[i:i+1])[0][0] * weight
            
            if selected:
                sim_to_selected = max(
                    cosine_similarity(doc_embs[i:i+1], doc_embs[selected])[0]
                )
            else:
                sim_to_selected = 0
            
            score = lambda_ * sim_to_query - (1 - lambda_) * sim_to_selected
            scores.append(score)

        best = candidates[np.argmax(scores)]
        selected.append(best)
        candidates.remove(best)

    return selected


def relative_news(temporal, embedding, type, host, port, collect_name, top_k=5):    
    valid_dates = temporal['timestep'].unique().tolist()
    
    response = search_news_by_embedding(host, port, collect_name, embedding, valid_dates)
    results = response["result"]
    
    doc_embeddings = [r["vector"] for r in results]
    docs = [r["payload"] for r in results]
    
    temporal_dict = dict(zip(temporal['timestep'], temporal['temporal_importance']))

    doc_dates = [r['payload']['collect_date'] for r in results]

    selected_idx = mmr_weighted(
        query_emb=embedding, 
        doc_embs=doc_embeddings, 
        doc_dates=doc_dates, 
        temporal_dict=temporal_dict,
        lambda_=0.8,
        top_k=top_k
    )
    
    final_docs = [docs[i] for i in selected_idx]
    
    selected_news = []
    for doc in final_docs:
        selected_news.append({
            "title": doc["title"],
            "description": doc["description"],
            "meta_site": doc["meta_site"],
            "url": doc["url"],
            "type": type,
            "publish_date": doc["publish_date"],
            "collect_date": doc["collect_date"]
        })

    return selected_news


def fetch_ticker_to_product_id(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT ticker, product_id FROM product")
    mapping = {row[0]: row[1] for row in cursor.fetchall()}
    cursor.close()
    return mapping


def prepare_insert_data(news_list, ticker_to_pid, base_date):
    insert_data = []
    for news in news_list:
        ticker = TYPE_TO_TICKER.get(news["type"].lower())
        if ticker is None or ticker not in ticker_to_pid:
            print(f"Unknown or unmapped type: {news['type']}, skipping...")
            continue

        product_id = ticker_to_pid[ticker]
        published_at = datetime.strptime(news["publish_date"], "%Y-%m-%d %H:%M:%S.%f")
        
        insert_data.append((
            product_id,
            news["title"],
            news["url"],
            news["meta_site"],
            published_at,
            base_date
        ))
    return insert_data


def insert_news(conn, insert_data):
    if not insert_data:
        print("No data to insert.")
        return
    
    cursor = conn.cursor()
    cursor.executemany(
        """
        INSERT INTO news (product_id, title, news_url, site_name, published_at, base_date)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        insert_data
    )
    conn.commit()
    cursor.close()
    print(f"Inserted {len(insert_data)} news items.")


def main():
    args = parse_args()

    load_dotenv()
    
    temporal_5 = pd.read_csv(f"{args.type}_{args.base_date}_tft_eval/w5/interpretations/{args.base_date}/temporal_importance.csv")
    temporal_20 = pd.read_csv(f"{args.type}_{args.base_date}_tft_eval/w20/interpretations/{args.base_date}/temporal_importance.csv")
    temporal_60 = pd.read_csv(f"{args.type}_{args.base_date}_tft_eval/w60/interpretations/{args.base_date}/temporal_importance.csv")
    
    temporal = pd.concat([temporal_5, temporal_20, temporal_60])
    temporal = temporal.groupby("timestep")["temporal_importance"].mean().reset_index()

    client = connect_vector_db(os.getenv('VECTOR_DB_HOST'), os.getenv('VECTOR_DB_PORT'))
    
    temporal["embedding"] = None
    for idx, row in temporal.iterrows():
        date = row["timestep"]
        vectors = fetch_embeddings_by_date(client, os.getenv('VECTOR_DB_NAME'), date, TYPE_TO_TYPES.get(args.type.lower()))
        pooled = mean_pool(vectors)

        if pooled is not None:
            temporal.at[idx, "embedding"] = pooled
            
    final_embedding = [float(x) for x in np.average(
        np.stack(temporal["embedding"].dropna().apply(np.array)),
        axis=0,
        weights=temporal["temporal_importance"].dropna()
    )]
    
    selected_news = relative_news(temporal, final_embedding, args.type, os.getenv('VECTOR_DB_HOST'), os.getenv('VECTOR_DB_PORT'), os.getenv('VECTOR_DB_NAME'))  
    
    conn = mysql.connector.connect(
        host=os.getenv('DB_HOST'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASS'),
        database=os.getenv('DB_NAME'),
        connect_timeout=10
    )
    ticker_to_pid = fetch_ticker_to_product_id(conn)
    insert_data = prepare_insert_data(selected_news, ticker_to_pid, args.base_date)
    insert_news(conn, insert_data)
    conn.close()
    # 지금은 웹 연결하려고 DB에만 저장하는데 prompt로 쓰려면 다른데에도 저장할 필요 있을듯? 아니면 바로 프롬포트화 시켜서 넘기던가


if __name__ == "__main__":
    main()