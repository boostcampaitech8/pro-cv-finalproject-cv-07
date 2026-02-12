import os
import argparse
from dotenv import load_dotenv

import boto3
import json

import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity

import json
from datetime import datetime, date


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
        default="2026-02-01"
    )
    parser.add_argument(
        "--type",
        type=str,
        default="gold",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="내일 금 가격 떨어질까? 너무불안해",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--distance-weight",
        type=int,
        default=0.8,
    )
    parser.add_argument(
        "--final-k",
        type=int,
        default=20
    )
    
    return parser.parse_args()


def prompt_embedding(input_text, dimensions, access_key_id, secret_access_key):
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
    
    embedding = titan_embeddings_v2(input_text, dimensions=dimensions, normalize=True)
    return embedding


def fetch_triples_from_neo4j(query_embedding, k, n, distance_weight, types, GRAPH_DB_URI, GRAPH_DB_AUTH):
    PIPELINE_QUERY = f"""
WITH
    $queryEmbedding AS queryEmbedding,
    $k AS k,
    $distance_weight AS distance_weight

CALL db.index.vector.queryNodes(
    'triple_embedding_index',
    k,
    queryEmbedding
)
YIELD node AS t0, score AS cosine_score

MATCH path =
    (t0)
    -[:HAS_SUBJECT|HAS_OBJECT|MENTIONS]-
    ()-
    [:HAS_SUBJECT|HAS_OBJECT|MENTIONS]-
    (t:TRIPLE)

WHERE t <> t0
    AND length(path) <= 2

MATCH (t)-[:MENTIONS]-(a:ARTICLE)
WHERE a.type IN $types

WITH
    t,
    cosine_score * (distance_weight ^ (length(path) - 1)) AS relevance

WITH
    t,
    max(relevance) AS relevance

MATCH (t)-[:MENTIONS]-(a:ARTICLE)
WHERE a.type IN $types

WITH
    t,
    relevance,
    a
ORDER BY a.publish_date DESC

WITH
    t,
    relevance,
    collect(a)[0] AS a

RETURN
    t.triple_id  AS triple_id,
    t.subject    AS subject,
    t.predicate  AS predicate,
    t.object     AS object,
    t.embedding  AS embedding,
    relevance,
    a.publish_date AS publish_date,
    a.meta_site   AS meta_site

ORDER BY relevance DESC
LIMIT {n};"""
    
    with GraphDatabase.driver(GRAPH_DB_URI, auth=GRAPH_DB_AUTH) as driver:
        driver.verify_connectivity()
        with driver.session() as session:
            result = session.run(
                PIPELINE_QUERY,
                queryEmbedding=query_embedding,
                k=k,
                distance_weight=distance_weight,
                types=types
            )
            return [dict(r) for r in result]


def time_decay(publish_date, base_date, alpha=0.01):
    base_date = pd.to_datetime(base_date, utc=True, errors="coerce").date()
    days = (base_date - publish_date).days
    return np.exp(-alpha * days)


def compute_time_aware_relevance(triples, base_date, alpha=0.01):
    relevances = []

    for t in triples:
        base_rel = t["relevance"]
        publish_date = pd.to_datetime(t["publish_date"], utc=True, errors="coerce").date()

        tw = time_decay(publish_date, base_date, alpha=alpha)
        relevances.append(base_rel * tw)

    return relevances


def mmr_time_aware(
    triples,
    base_date, 
    lambda_=0.8,
    top_k=20,
    alpha=0.01
):
    embeddings = np.array([t["embedding"] for t in triples])
    base_rels = compute_time_aware_relevance(triples, base_date, alpha)

    selected = []
    candidates = list(range(len(triples)))

    while len(selected) < top_k and candidates:
        scores = []

        for i in candidates:
            rel = base_rels[i]

            if selected:
                sim = max(
                    cosine_similarity(
                        embeddings[i:i+1],
                        embeddings[selected]
                    )[0]
                )
            else:
                sim = 0

            score = lambda_ * rel - (1 - lambda_) * sim
            scores.append(score)

        best = candidates[np.argmax(scores)]
        selected.append(best)
        candidates.remove(best)

    return [triples[i] for i in selected]


def json_serializer(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def main():
    args = parse_args()

    load_dotenv()

    embedding = prompt_embedding(args.prompt, 1024, os.getenv("AWS_ACCESS_KEY_ID"), os.getenv("AWS_SECRET_ACCESS_KEY"))
    
    URI = os.getenv("GRAPH_DB_URI")
    AUTH = (os.getenv("GRAPH_DB_USER"), os.getenv("GRAPH_DB_PASS"))

    selected = fetch_triples_from_neo4j(embedding, args.k, args.n, args.distance_weight, TYPE_TO_TYPES.get(args.type.lower()), URI, AUTH)
    final_triples = mmr_time_aware(selected, args.base_date, lambda_=0.8, top_k=args.final_k)
    
    for t in final_triples:
        print(
            f"{t['subject']} - {t['predicate']} - {t['object']} "
            f"({t['meta_site']}, {t['publish_date']})"
            f"(relevance={t['relevance']:.3f})"
        )    

    with open("final_triples.json", "w", encoding="utf-8") as f:
        json.dump(
            final_triples,
            f,
            ensure_ascii=False,
            indent=2,
            default=json_serializer
        )
    

if __name__ == "__main__":
    main()
