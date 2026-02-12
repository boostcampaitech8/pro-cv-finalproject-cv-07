import os
import logging
import json

import boto3
import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity
from datetime import date

logger = logging.getLogger(__name__)

# ticker → Neo4j article type 필터
TICKER_TO_TYPES = {
    "ZC=F": ["corn", "crop-general"],
    "ZW=F": ["wheat", "crop-general"],
    "ZS=F": ["soybean", "crop-general"],
    "GC=F": ["gold"],
    "SI=F": ["sliver"],
    "HG=F": ["copper"],
}


def _get_bedrock_client():
    return boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )


def embed_text(text: str, dimensions: int = 1024) -> list:
    """Amazon Titan Embed v2로 텍스트 임베딩"""
    client = _get_bedrock_client()
    body = json.dumps({
        "inputText": text,
        "dimensions": dimensions,
        "normalize": True,
    })
    response = client.invoke_model(
        body=body,
        modelId="amazon.titan-embed-text-v2:0",
        accept="application/json",
        contentType="application/json",
    )
    return json.loads(response["body"].read())["embedding"]


def fetch_triples(query_embedding: list, types: list, k: int = 5, n: int = 100, distance_weight: float = 0.8) -> list:
    """Neo4j에서 벡터 유사도 + 2-hop 그래프 확장으로 트리플 조회"""
    uri = os.getenv("NEO4J_URI")
    auth = (os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))

    query = f"""
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
    a.meta_site   AS meta_site,
    a.url         AS url,
    a.title       AS title

ORDER BY relevance DESC
LIMIT {n};"""

    with GraphDatabase.driver(uri, auth=auth) as driver:
        driver.verify_connectivity()
        with driver.session() as session:
            result = session.run(
                query,
                queryEmbedding=query_embedding,
                k=k,
                distance_weight=distance_weight,
                types=types,
            )
            return [dict(r) for r in result]


def _time_decay(publish_date, base_date, alpha=0.01):
    base_dt = pd.to_datetime(base_date, utc=True, errors="coerce").date()
    pub_dt = pd.to_datetime(publish_date, utc=True, errors="coerce").date()
    days = (base_dt - pub_dt).days
    return np.exp(-alpha * max(days, 0))


def mmr_select(triples: list, base_date=None, lambda_: float = 0.8, top_k: int = 20, alpha: float = 0.01) -> list:
    """시간 가중 MMR로 다양성+관련성 균형 있는 트리플 선정"""
    if not triples:
        return []

    if base_date is None:
        base_date = date.today().isoformat()

    embeddings = np.array([t["embedding"] for t in triples])
    base_rels = [
        t["relevance"] * _time_decay(t["publish_date"], base_date, alpha)
        for t in triples
    ]

    selected = []
    candidates = list(range(len(triples)))

    while len(selected) < top_k and candidates:
        scores = []
        for i in candidates:
            rel = base_rels[i]
            if selected:
                sim = max(cosine_similarity(embeddings[i:i+1], embeddings[selected])[0])
            else:
                sim = 0
            scores.append(lambda_ * rel - (1 - lambda_) * sim)

        best = candidates[np.argmax(scores)]
        selected.append(best)
        candidates.remove(best)

    return [triples[i] for i in selected]


def retrieve_graph_context(user_message: str, ticker: str) -> dict:
    """사용자 질문으로 Graph RAG context를 조회하여 텍스트와 출처 목록 반환"""
    types = TICKER_TO_TYPES.get(ticker)
    if not types:
        logger.warning(f"[GraphRAG] ticker '{ticker}'에 대한 타입 매핑 없음")
        return {"context": "", "sources": []}

    try:
        logger.info(f"[GraphRAG] 임베딩 시작: ticker={ticker}, types={types}")
        embedding = embed_text(user_message)

        logger.info(f"[GraphRAG] Neo4j 트리플 조회 시작")
        raw_triples = fetch_triples(embedding, types)
        logger.info(f"[GraphRAG] 조회된 트리플 수: {len(raw_triples)}")

        if not raw_triples:
            return {"context": "", "sources": []}

        final_triples = mmr_select(raw_triples, top_k=20)
        logger.info(f"[GraphRAG] MMR 선정 트리플 수: {len(final_triples)}")

        lines = []
        for t in final_triples:
            lines.append(f"- {t['subject']} | {t['predicate']} | {t['object']} (출처: {t['meta_site']})")

        # 출처 중복 제거 (url 기준)
        seen_urls = set()
        sources = []
        for t in final_triples:
            url = t.get("url") or ""
            title = t.get("title") or t.get("meta_site") or ""
            pub_date = t.get("publish_date")
            if pub_date:
                pub_date = pd.to_datetime(pub_date, utc=True, errors="coerce")
                pub_date = pub_date.strftime("%Y-%m-%d") if pub_date else ""

            key = url or title
            if key and key not in seen_urls:
                seen_urls.add(key)
                sources.append({
                    "title": title,
                    "url": url,
                    "publish_date": pub_date or "",
                    "site": t.get("meta_site") or "",
                })

        return {"context": "\n".join(lines), "sources": sources[:10]}
    except Exception as e:
        logger.error(f"[GraphRAG] 조회 실패: {e}", exc_info=True)
        return {"context": "", "sources": []}
