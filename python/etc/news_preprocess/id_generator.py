#!/usr/bin/env python3
"""
고유 ID 생성 모듈
- UUID5를 사용하여 결정론적(deterministic) ID 생성
- 동일한 입력에 대해 항상 동일한 ID 반환
"""

import uuid
import pandas as pd


def generate_global_id(content: str) -> str:
    """
    문자열을 기반으로 결정론적 UUID 생성
    동일한 content에 대해 항상 동일한 UUID 반환
    """
    if not content:
        return str(uuid.uuid4())
    return str(uuid.uuid5(uuid.NAMESPACE_URL, str(content)))


def generate_article_id(row: pd.Series) -> str:
    """
    뉴스 기사 ID 생성
    - 1순위: doc_url (가장 확실한 고유키)
    - 2순위: title + publish_date (URL 없을 때 대체)
    """
    if 'doc_url' in row and pd.notna(row['doc_url']) and str(row['doc_url']).strip():
        unique_key = str(row['doc_url']).strip()
    else:
        date_str = str(row.get('publish_date', '')).split(' ')[0]
        unique_key = f"{str(row.get('title', '')).strip()}_{date_str}"

    return generate_global_id(unique_key)


# 해시 기반
def generate_entity_id(entity_name: str) -> str:
    """
    Entity ID 생성 (정규화 적용)
    - 소문자 변환 + 공백 제거
    """
    if not entity_name:
        return generate_global_id("unknown_entity")

    clean_name = str(entity_name).strip().lower()
    return generate_global_id(clean_name)


def generate_triple_id(subject: str, predicate: str, object_: str) -> str:
    """
    Triple ID 생성
    - Subject|Predicate|Object 형태로 결합
    """
    s = str(subject).strip().lower()
    p = str(predicate).strip().lower()
    o = str(object_).strip().lower()

    composite_key = f"{s}|{p}|{o}"
    return generate_global_id(composite_key)


def add_ids_to_dataframes(
    df: pd.DataFrame,
    entities_df: pd.DataFrame,
    triples_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    모든 DataFrame에 고유 ID 추가

    Args:
        df: 뉴스 DataFrame
        entities_df: 엔티티 DataFrame (article_id, entity_value)
        triples_df: 트리플 DataFrame (article_id, subject, predicate, object)

    Returns:
        (df, entities_df, triples_df) - id 컬럼이 추가된 DataFrame 튜플
    """
    df = df.copy()
    entities_df = entities_df.copy()
    triples_df = triples_df.copy()

    # Article ID 생성
    df['id'] = df.apply(generate_article_id, axis=1)

    # article_id(index) -> article uuid 매핑 생성
    article_id_map = df['id'].to_dict()

    # Entity ID 생성 + article_uuid 매핑
    if not entities_df.empty:
        entities_df['id'] = entities_df['entity_value'].apply(generate_entity_id)
        entities_df['article_uuid'] = entities_df['article_id'].map(article_id_map)

    # Triple ID 생성 + article_uuid 매핑
    if not triples_df.empty:
        triples_df['id'] = triples_df.apply(
            lambda x: generate_triple_id(x['subject'], x['predicate'], x['object']),
            axis=1
        )
        triples_df['article_uuid'] = triples_df['article_id'].map(article_id_map)

    return df, entities_df, triples_df


if __name__ == "__main__":
    # 테스트
    print("=== ID Generator Test ===")

    # Article ID 테스트
    row_sample = pd.Series({
        'doc_url': 'http://news.com/article/123',
        'title': 'Samsung Launches Galaxy',
        'publish_date': '2025-02-05 10:00:00'
    })
    art_id = generate_article_id(row_sample)
    print(f"Article ID: {art_id}")

    # Entity ID 테스트 (정규화 확인)
    ent_id_1 = generate_entity_id("Samsung Electronics")
    ent_id_2 = generate_entity_id("samsung electronics ")
    print(f"Entity ID Match: {ent_id_1 == ent_id_2}")  # True

    # Triple ID 테스트
    trip_id = generate_triple_id("Samsung", "Announced", "Galaxy S24")
    print(f"Triple ID: {trip_id}")
