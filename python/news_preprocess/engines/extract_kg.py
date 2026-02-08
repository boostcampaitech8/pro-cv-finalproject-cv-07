#!/usr/bin/env python3
"""
Knowledge Graph 추출 모듈
- LLM을 사용하여 뉴스에서 Entities와 Triples 추출
- status_judge.py와 동일한 모델 사용 (싱글톤)
"""

import os
import time
import pandas as pd
import torch
import json
import re
from pathlib import Path
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# status_judge.py의 모델 로딩 함수 재사용
import sys
_NEWS_PREPROCESS_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_NEWS_PREPROCESS_ROOT))
from engines.status_judge import load_model, DEFAULT_MODEL

from dotenv import load_dotenv

# .env 로드
_PYTHON_ROOT = _NEWS_PREPROCESS_ROOT.parent  # python/
_REPO_ROOT = _PYTHON_ROOT.parent
load_dotenv(_REPO_ROOT / ".env")

DEFAULT_NANO_MODEL = "openai:gpt-4.1-mini"


# ============================================================
# 프롬프트 템플릿
# ============================================================
SYSTEM_PROMPT = """You are an expert AI for Knowledge Graph Extraction.
Your task is to extract **Entities (Nodes)** and **Relationships (Edges)** from the input text to build a Knowledge Graph.

**STRICT GUIDELINES:**
1. **Nodes (Entities)**:
   - Extract key entities (Person, Organization, Location, Concept, Event, Date, Quantity, etc.).
   - Assign a unique `id` (e.g., "n1", "n2") to each node.
   - `name`: Use the exact text or a concise atomic name (e.g., "Indian farmers", NOT "Indian farmers in the field").

2. **Relationships (Edges)**:
   - Identify meaningful connections between nodes.
   - **Relation Type Style**: Use **natural language phrases** (verbs/prepositions) found in or inferred from the text.
     - **DO NOT** use generic/technical tags like "LOCATED_IN", "PART_OF", "HAS_ENTITY".
     - **DO** use descriptive phrases like: "is located in", "affected by", "warns against", "soared due to", "reluctant to bet on".
   - Keep relation types **lowercase**.

3. **Output Format**:
   - Return **ONLY** a valid JSON object. Do not include any explanations or markdown.
   - JSON Structure:
     {"nodes": [{"id": "n1", "name": "..."}], "relationships": [{"source_id": "n1", "type": "...", "target_id": "n2"}]}"""

USER_TEMPLATE = """Title: {title}
Description: {description}

Extract entities and relationships as JSON."""

# ============================================================
# Predicate-only verification prompt (keep nodes/ids)
# ============================================================
PREDICATE_VERIFY_SYSTEM_PROMPT = """You are a predicate verifier for a knowledge graph.

You will be given TEXT, NODES, and RELATIONSHIPS.
Your job is to verify and improve each relationship "type" so it faithfully represents a fact stated in TEXT.

Rules:
1) Do NOT change nodes. Return the exact same nodes list.
2) Do NOT change relationship source_id or target_id. Never produce a self-loop (source_id == target_id).
3) Each relationship "type" must be a short lowercase verb phrase (2-5 words MAXIMUM) that closely reflects wording in TEXT.
   - Prefer copying phrases from TEXT, but minor rephrasing for grammatical correctness is allowed.
   - The predicate must preserve the factual meaning of the original relation.
   - Do not embed numeric values or entity names (subject, object) in the predicate
4) DROP a relationship ONLY IF the relation is clearly not stated or implied in TEXT.
5) Do NOT reduce predicates to single prepositions or function words (e.g. "of", "to", "with", "are", "is").
6) Do NOT add new nodes or relationships.

Output ONLY valid JSON in the same schema:
{"nodes":[{"id":"...","name":"..."}],
 "relationships":[{"source_id":"...","type":"...","target_id":"..."}]}"""

PREDICATE_VERIFY_USER_TEMPLATE = """TEXT:
Title: {title}
Description: {description}

NODES (JSON):
{nodes_json}

RELATIONSHIPS (JSON):
{rels_json}

Return JSON only."""


# ============================================================
# 헬퍼 함수
# ============================================================
def _build_messages(title: str, description: str) -> list:
    """Chat 형식 메시지 생성"""
    user_content = USER_TEMPLATE.format(
        title=str(title)[:200] if pd.notna(title) else "",
        description=str(description)[:500] if pd.notna(description) else ""
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]

def _build_verify_messages(title: str, description: str, nodes_json: str, rels_json: str) -> list:
    user_content = PREDICATE_VERIFY_USER_TEMPLATE.format(
        title=str(title)[:200] if pd.notna(title) else "",
        description=str(description)[:800] if pd.notna(description) else "",
        nodes_json=nodes_json,
        rels_json=rels_json,
    )
    return [
        {"role": "system", "content": PREDICATE_VERIFY_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _parse_kg_response(response: str) -> dict:
    """응답에서 JSON 파싱하여 nodes, relationships 추출"""
    # <think> 태그 제거
    clean_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    clean_response = re.sub(r'<think>.*', '', clean_response, flags=re.DOTALL).strip()

    try:
        # JSON 블록 찾기
        json_match = re.search(r"\{.*\}", clean_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        else:
            # 전체가 JSON인 경우
            return json.loads(clean_response)
    except json.JSONDecodeError:
        return {"nodes": [], "relationships": []}


# ============================================================
# KG 추출 함수
# ============================================================
def extract_kg_single(title: str, description: str, model_name: str = None) -> dict:
    """
    단일 기사에서 KG 추출

    Returns:
        {"nodes": [...], "relationships": [...]}
    """
    model, tokenizer = load_model(model_name or DEFAULT_MODEL)
    messages = _build_messages(title, description)

    if hasattr(tokenizer, 'apply_chat_template'):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = f"System: {SYSTEM_PROMPT}\n\nUser: {messages[1]['content']}\n\nAssistant:"

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return _parse_kg_response(response)

def _openai_client():
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("OpenAI Python SDK not installed. Try: pip install openai") from e
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Put it in .env or environment.")
    return OpenAI(api_key=api_key)


def _openai_model_id(model_name: str) -> str:
    if model_name.lower().startswith("openai:"):
        return model_name.split(":", 1)[1].strip()
    return model_name


def verify_predicates(
    title: str,
    description: str,
    kg: dict,
    model_name: str = None,
) -> dict:
    """Keep nodes/ids fixed and re-align relationship predicate to text.
    Uses OpenAI API (nano/mini model) for verification."""
    nodes = kg.get("nodes", []) if isinstance(kg, dict) else []
    rels = kg.get("relationships", []) if isinstance(kg, dict) else []

    # If no relationships, return early
    if not rels:
        return {"nodes": nodes, "relationships": []}

    nano = model_name or DEFAULT_NANO_MODEL
    client = _openai_client()

    nodes_json = json.dumps(nodes, ensure_ascii=False)
    rels_json = json.dumps(rels, ensure_ascii=False)
    messages = _build_verify_messages(title, description, nodes_json, rels_json)

    for attempt in range(3):
        try:
            response = client.responses.create(
                model=_openai_model_id(nano),
                input=messages,
                max_output_tokens=768,
                temperature=0,
            )
            text = getattr(response, "output_text", None)
            if text is None:
                try:
                    text = response.output[0].content[0].text
                except Exception:
                    text = str(response)
            verified = _parse_kg_response(text)
            break
        except Exception as e:
            if attempt >= 2:
                print(f"  [verify_predicates] API failed after 3 attempts: {e}")
                return {"nodes": nodes, "relationships": rels}
            time.sleep(2 ** attempt)

    # Safety: keep original nodes, filter relationships to valid ids only
    node_ids = {n.get("id") for n in nodes if isinstance(n, dict)}
    cleaned_rels = []
    for rel in verified.get("relationships", []) if isinstance(verified, dict) else []:
        if not isinstance(rel, dict):
            continue
        src = rel.get("source_id")
        tgt = rel.get("target_id")
        typ = rel.get("type")
        if src in node_ids and tgt in node_ids and src != tgt and typ:
            cleaned_rels.append({
                "source_id": src,
                "type": str(typ).strip().lower(),
                "target_id": tgt,
            })
    return {"nodes": nodes, "relationships": cleaned_rels}


def extract_kg_batch(
    df: pd.DataFrame,
    model_name: str = DEFAULT_MODEL,
    title_col: str = "title",
    desc_col: str = "description",
    verify_predicate_only: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    배치 KG 추출

    Args:
        df: DataFrame (title, description 컬럼 필요)
        model_name: 사용할 모델명
        title_col: 제목 컬럼명
        desc_col: 설명 컬럼명

    Returns:
        (entities_df, triples_df) 튜플
        - entities_df: article_id, entity_value
        - triples_df: article_id, subject, predicate, object
    """
    print(f"\n[Knowledge Graph Extraction]")
    print(f"  Processing {len(df)} articles...")

    # Step 1: 로컬 LLM으로 KG 추출 (순차 - GPU 사용)
    raw_results = []  # list of (index, title, description, kg)
    for i, row in df.iterrows():
        try:
            kg = extract_kg_single(
                row.get(title_col, ""),
                row.get(desc_col, ""),
                model_name
            )
            raw_results.append((i, row.get(title_col, ""), row.get(desc_col, ""), kg))
        except Exception as e:
            print(f"    Error on row {i}: {e}")
            continue

        if (len(raw_results)) % 10 == 0:
            print(f"    Extracted {len(raw_results)}/{len(df)}")

    # Step 2: OpenAI API로 predicate 검증 (병렬 4개)
    if verify_predicate_only and raw_results:
        print(f"  Verifying predicates ({len(raw_results)} articles, 4 workers)...")

        def _verify_one(item):
            idx, title, desc, kg = item
            try:
                verified = verify_predicates(title, desc, kg)
                return (idx, verified)
            except Exception as e:
                print(f"    Verify error on row {idx}: {e}")
                return (idx, kg)

        verified_map = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_verify_one, item): item[0] for item in raw_results}
            for future in as_completed(futures):
                idx, verified_kg = future.result()
                verified_map[idx] = verified_kg
                if len(verified_map) % 10 == 0:
                    print(f"    Verified {len(verified_map)}/{len(raw_results)}")

        # raw_results의 kg를 verified로 교체
        raw_results = [(idx, t, d, verified_map.get(idx, kg)) for idx, t, d, kg in raw_results]

    # Step 3: 결과 수집
    entities_data = []
    triples_data = []

    for idx, title, desc, kg in raw_results:
        article_id = idx
        id_to_name = {}

        for node in kg.get("nodes", []):
            node_id = node.get("id")
            name = node.get("name")
            if node_id and name:
                id_to_name[node_id] = name
                entities_data.append({
                    "article_id": article_id,
                    "entity_value": name
                })

        for rel in kg.get("relationships", []):
            src_id = rel.get("source_id")
            tgt_id = rel.get("target_id")
            rel_type = rel.get("type")

            if src_id in id_to_name and tgt_id in id_to_name:
                triples_data.append({
                    "article_id": article_id,
                    "subject": id_to_name[src_id],
                    "predicate": rel_type,
                    "object": id_to_name[tgt_id]
                })

    entities_df = pd.DataFrame(entities_data)
    triples_df = pd.DataFrame(triples_data)

    # 빈 DataFrame인 경우 컬럼 추가
    if entities_df.empty:
        entities_df = pd.DataFrame(columns=['article_id', 'entity_value'])
    if triples_df.empty:
        triples_df = pd.DataFrame(columns=['article_id', 'subject', 'predicate', 'object'])

    print(f"  Extracted {len(entities_df)} entities, {len(triples_df)} triples")

    return entities_df, triples_df


# ============================================================
# 메인 (테스트용)
# ============================================================
if __name__ == "__main__":
    from status_judge import unload_model

    # 테스트
    test_title = "Soybean prices surge as drought hits Argentina"
    test_desc = "Argentine farmers face significant crop losses due to prolonged dry conditions. USDA reports lower production estimates."

    print("Testing KG extraction...")
    kg = extract_kg_single(test_title, test_desc)
    kg = verify_predicates(test_title, test_desc, kg)
    print(f"Nodes: {kg.get('nodes', [])}")
    print(f"Relationships: {kg.get('relationships', [])}")

    unload_model()
