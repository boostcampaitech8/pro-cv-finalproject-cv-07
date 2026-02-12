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
CONNECTIONS_DIR = Path(__file__).resolve().parent     # connections/
AIRFLOW_ROOT = CONNECTIONS_DIR.parent                 # airflow/
REPO_ROOT = AIRFLOW_ROOT.parent                       # pro-cv-finalproject-cv-07/
sys.path.insert(0, str(AIRFLOW_ROOT / 'plugins' / 'article'))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / '.env')

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
    triple이 None인 레코드는 ARTICLE 노드만 생성
    """
    with_triple = [r for r in batch_records if r.get("triple") is not None]
    without_triple = [r for r in batch_records if r.get("triple") is None]

    # ARTICLE만 저장 (triple 없는 경우)
    if without_triple:
        tx.run(
            """
            UNWIND $batch AS record
            MERGE (a:ARTICLE {article_id: record.article.article_id})
            SET a.type = record.article.type,
                a.url = record.article.doc_url,
                a.publish_date = record.article.publish_date,
                a.collect_date = record.article.collect_date,
                a.title = record.article.title,
                a.meta_site = record.article.meta_site,
                a.description = record.article.description,
                a.embedding = record.article.embedding
            """,
            batch=without_triple
        )

    # ARTICLE + TRIPLE + ENTITY 저장
    if with_triple:
        tx.run(
            """
            UNWIND $batch AS record

            MERGE (a:ARTICLE {article_id: record.article.article_id})
            SET a.type = record.article.type,
                a.url = record.article.doc_url,
                a.publish_date = record.article.publish_date,
                a.collect_date = record.article.collect_date,
                a.title = record.article.title,
                a.meta_site = record.article.meta_site,
                a.description = record.article.description,
                a.embedding = record.article.embedding

            MERGE (t:TRIPLE {triple_id: record.triple.hash_id})
            SET t.subject = record.triple.subject,
                t.predicate = record.triple.predicate,
                t.object = record.triple.object,
                t.embedding = record.triple.embedding

            MERGE (a)-[:MENTIONS]->(t)

            FOREACH (subj IN record.subjects |
                MERGE (s:ENTITY {entity_id: subj.hash_id})
                SET s.entity_text = subj.entity_text,
                    s.embedding = subj.embedding
                MERGE (t)-[:HAS_SUBJECT]->(s)
                MERGE (a)-[:CONTAINS_ENTITY]->(s)
            )

            FOREACH (obj IN record.objects |
                MERGE (o:ENTITY {entity_id: obj.hash_id})
                SET o.entity_text = obj.entity_text,
                    o.embedding = obj.embedding
                MERGE (t)-[:HAS_OBJECT]->(o)
                MERGE (a)-[:CONTAINS_ENTITY]->(o)
            )
            """,
            batch=with_triple
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
            # 트리플이 없으면 기사 노드만 별도 저장 (TRIPLE MERGE 스킵)
            records.append({
                "article": article_info,
                "triple": None,
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
if __name__ == "__main__":
    try:
        driver = get_driver()
        driver.verify_connectivity()
        print("Neo4j connection OK")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        close_driver()
