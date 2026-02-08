#!/usr/bin/env python3
"""
Neo4j GraphDB 삽입 유틸리티

main.py에서 사용:
    from graphdb_utils import save_to_graphdb
    save_to_graphdb(df, entities_df, triples_df)
"""

import os
import sys
from pathlib import Path

# 경로 설정
NEWS_PREPROCESS_ROOT = Path(__file__).resolve().parent.parent  # news_preprocess/
PYTHON_ROOT = NEWS_PREPROCESS_ROOT.parent  # python/
REPO_ROOT = PYTHON_ROOT.parent
sys.path.insert(0, str(PYTHON_ROOT))
sys.path.insert(0, str(NEWS_PREPROCESS_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from tqdm import tqdm

from crawler_config import KEYWORD_MAPPING


# ============================================================
# Neo4j 연결
# ============================================================
_driver = None


def get_driver():
    """Neo4j 드라이버 싱글톤"""
    global _driver

    if _driver is None:
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD")

        if not uri or not password:
            raise ValueError(
                "NEO4J_URI, NEO4J_PASSWORD 환경변수 필요\n"
                ".env 파일에 추가하세요"
            )

        _driver = GraphDatabase.driver(uri, auth=(user, password))
        _driver.verify_connectivity()

    return _driver


def close_driver():
    """드라이버 연결 종료"""
    global _driver
    if _driver:
        _driver.close()
        _driver = None


# ============================================================
# 배치 삽입 함수
# ============================================================
BATCH_SIZE = 100


def _safe_embedding(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    # pandas NA/NaN 처리 (스칼라에만 적용)
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    return value


def _create_nodes_and_relationships_batch(tx, batch_records):
    """
    배치로 노드와 관계 생성

    Cypher 쿼리:
    - ARTICLE 노드 생성/업데이트
    - TRIPLE 노드 생성/업데이트
    - ARTICLE -> TRIPLE (MENTIONS) 관계
    - TRIPLE -> ENTITY (HAS_SUBJECT, HAS_OBJECT) 관계
    - ARTICLE -> ENTITY (CONTAINS_ENTITY) 관계
    """
    tx.run(
        """
        UNWIND $batch AS record

        // ARTICLE 노드
        MERGE (a:ARTICLE {article_id: record.article.article_id})
        SET a.type = record.article.type,
            a.url = record.article.doc_url,
            a.publish_date = record.article.publish_date,
            a.collect_date = record.article.collect_date,
            a.title = record.article.title,
            a.meta_site = record.article.meta_site,
            a.description = record.article.description,
            a.embedding = record.article.embedding

        // TRIPLE 노드
        MERGE (t:TRIPLE {triple_id: record.triple.hash_id})
        SET t.subject = record.triple.subject,
            t.predicate = record.triple.predicate,
            t.object = record.triple.object,
            t.embedding = record.triple.embedding

        // ARTICLE -> TRIPLE 관계
        MERGE (a)-[:MENTIONS]->(t)

        // SUBJECT ENTITY 노드 및 관계
        FOREACH (subj IN record.subjects |
            MERGE (s:ENTITY {entity_id: subj.hash_id})
            SET s.entity_text = subj.entity_text,
                s.embedding = subj.embedding
            MERGE (t)-[:HAS_SUBJECT]->(s)
            MERGE (a)-[:CONTAINS_ENTITY]->(s)
        )

        // OBJECT ENTITY 노드 및 관계
        FOREACH (obj IN record.objects |
            MERGE (o:ENTITY {entity_id: obj.hash_id})
            SET o.entity_text = obj.entity_text,
                o.embedding = obj.embedding
            MERGE (t)-[:HAS_OBJECT]->(o)
            MERGE (a)-[:CONTAINS_ENTITY]->(o)
        )
        """,
        batch=batch_records
    )


# ============================================================
# 데이터 변환
# ============================================================
def _match_entities_to_triple(
    triple_row: pd.Series,
    entities_df: pd.DataFrame,
    article_uuid: str,
    article_type: str = "",
) -> tuple[list, list]:
    """
    트리플의 subject/object와 매칭되는 엔티티 찾기

    Args:
        triple_row: 트리플 행 (subject, predicate, object 컬럼 포함)
        entities_df: 엔티티 DataFrame
        article_uuid: 기사 UUID (해당 기사의 엔티티만 검색)
        article_type: 기사 키워드 (entity type으로 사용)

    Returns:
        (subject_entities, object_entities)
    """
    article_entities = entities_df[entities_df['article_uuid'] == article_uuid]

    if article_entities.empty:
        return [], []

    subject = str(triple_row.get('subject', '')).strip().lower()
    obj = str(triple_row.get('object', '')).strip().lower()

    subject_entities = []
    object_entities = []

    for _, ent_row in article_entities.iterrows():
        entity_value = str(ent_row.get('entity_value', '')).strip().lower()

        # subject 매칭 (부분 문자열 포함)
        if entity_value and subject:
            if entity_value in subject or subject in entity_value:
                subject_entities.append({
                    "entity_id": ent_row.get('entity_uuid', ''),
                    "hash_id": ent_row.get('hash_id', ''),
                    "value": entity_value,
                    "type": article_type,
                    "embedding": _safe_embedding(ent_row.get('embedding', [])),
                })

        # object 매칭
        if entity_value and obj:
            if entity_value in obj or obj in entity_value:
                object_entities.append({
                    "entity_id": ent_row.get('entity_uuid', ''),
                    "hash_id": ent_row.get('hash_id', ''),
                    "value": entity_value,
                    "type": article_type,
                    "embedding": _safe_embedding(ent_row.get('embedding', [])),
                })

    return subject_entities, object_entities


def prepare_graphdb_records(
    df: pd.DataFrame,
    entities_df: pd.DataFrame,
    triples_df: pd.DataFrame
) -> list:
    """
    DataFrame들을 GraphDB 삽입용 레코드로 변환

    Args:
        df: 기사 DataFrame (article_uuid, publish_date, title, description, embedding)
        entities_df: 엔티티 DataFrame (entity_uuid, article_uuid, entity_value, entity_type, embedding)
        triples_df: 트리플 DataFrame (triple_uuid, article_uuid, subject, predicate, object, value, embedding)

    Returns:
        GraphDB 삽입용 레코드 리스트
    """
    records = []

    for _, article_row in tqdm(df.iterrows(), total=len(df), desc="Preparing GraphDB records"):
        article_uuid = article_row.get('article_uuid', '')
        raw_keyword = article_row.get('key_word', 'gold')
        article_type = KEYWORD_MAPPING.get(raw_keyword, raw_keyword)

        # 기사 정보
        article_info = {
            "article_id": article_uuid,
            "type": article_type,
            "doc_url": article_row.get('doc_url', ''),
            "publish_date": str(article_row.get('publish_date', ''))[:10],
            "collect_date": str(article_row.get('collect_date', ''))[:10],
            "title": article_row.get('title', ''),
            "meta_site": article_row.get('meta_site_name', ''),
            "description": article_row.get('description', ''),
            "embedding": _safe_embedding(article_row.get('embedding', [])),
        }

        # 해당 기사의 트리플들
        article_triples = triples_df[triples_df['article_uuid'] == article_uuid]

        if article_triples.empty:
            # 트리플이 없어도 기사는 저장
            records.append({
                "article": article_info,
                "triple": {
                    "triple_id": "",
                    "subject": "",
                    "predicate": "",
                    "object": "",
                    "value": "",
                    "type": "",
                    "embedding": [],
                },
                "subjects": [],
                "objects": [],
            })
            continue

        for _, triple_row in article_triples.iterrows():
            subject = str(triple_row.get('subject', '')).strip()
            predicate = str(triple_row.get('predicate', '')).strip()
            obj = str(triple_row.get('object', '')).strip()
            value = triple_row.get('value', '')

            triple_info = {
                "triple_id": triple_row.get('triple_uuid', ''),
                "hash_id": triple_row.get('hash_id', ''),
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "value": str(value),
                "type": article_type,
                "embedding": _safe_embedding(triple_row.get('embedding', [])),
            }

            # 엔티티 매칭
            subject_entities, object_entities = _match_entities_to_triple(
                triple_row, entities_df, article_uuid, article_type
            )

            records.append({
                "article": article_info,
                "triple": triple_info,
                "subjects": subject_entities,
                "objects": object_entities,
            })

    return records


# ============================================================
# 메인 저장 함수
# ============================================================
def save_to_graphdb(
    df: pd.DataFrame,
    entities_df: pd.DataFrame,
    triples_df: pd.DataFrame,
    batch_size: int = BATCH_SIZE
) -> int:
    """
    GraphDB에 데이터 저장

    Args:
        df: 기사 DataFrame
        entities_df: 엔티티 DataFrame
        triples_df: 트리플 DataFrame
        batch_size: 배치 크기

    Returns:
        저장된 레코드 수
    """
    print("\n[GraphDB Save]")

    # 데이터 검증
    if df.empty:
        print("  No articles to save")
        return 0

    # 필수 컬럼 확인
    required_cols = ['article_uuid']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"  Missing columns: {missing}")
        return 0

    # 레코드 준비
    records = prepare_graphdb_records(df, entities_df, triples_df)

    if not records:
        print("  No records to insert")
        return 0

    # 드라이버 연결
    driver = get_driver()

    # 배치 삽입
    total_inserted = 0
    with driver.session() as session:
        for i in tqdm(range(0, len(records), batch_size), desc="Inserting into Neo4j"):
            batch = records[i:i + batch_size]
            session.execute_write(_create_nodes_and_relationships_batch, batch)
            total_inserted += len(batch)

    print(f"  Inserted {total_inserted} records ({len(df)} articles)")

    return total_inserted


# ============================================================
# 유틸리티 함수
# ============================================================
def count_nodes():
    """각 노드 타입별 개수 조회"""
    driver = get_driver()

    with driver.session() as session:
        result = session.run("""
            MATCH (a:ARTICLE) WITH count(a) as articles
            MATCH (e:ENTITY) WITH articles, count(e) as entities
            MATCH (t:TRIPLE) WITH articles, entities, count(t) as triples
            RETURN articles, entities, triples
        """)
        record = result.single()
        return {
            "articles": record["articles"],
            "entities": record["entities"],
            "triples": record["triples"],
        }


def get_article_graph(article_uuid: str):
    """특정 기사의 그래프 조회"""
    driver = get_driver()

    with driver.session() as session:
        result = session.run("""
            MATCH (a:ARTICLE {article_uuid: $uuid})
            OPTIONAL MATCH (a)-[:MENTIONS]->(t:TRIPLE)
            OPTIONAL MATCH (t)-[:HAS_SUBJECT]->(s:ENTITY)
            OPTIONAL MATCH (t)-[:HAS_OBJECT]->(o:ENTITY)
            RETURN a, collect(DISTINCT t) as triples,
                   collect(DISTINCT s) as subjects,
                   collect(DISTINCT o) as objects
        """, uuid=article_uuid)
        return result.single()


if __name__ == "__main__":
    # 테스트: 노드 개수 조회
    try:
        counts = count_nodes()
        print("Node counts:")
        print(f"  Articles: {counts['articles']}")
        print(f"  Entities: {counts['entities']}")
        print(f"  Triples: {counts['triples']}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        close_driver()
