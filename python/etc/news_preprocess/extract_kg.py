#!/usr/bin/env python3
"""
Knowledge Graph 추출 모듈
- LLM을 사용하여 뉴스에서 Entities와 Triples 추출
- status_judge.py와 동일한 모델 사용 (싱글톤)
"""

import pandas as pd
import torch
import json
import re
from typing import Tuple

# status_judge.py의 모델 로딩 함수 재사용
from status_judge import load_model, DEFAULT_MODEL


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


def extract_kg_batch(
    df: pd.DataFrame,
    model_name: str = DEFAULT_MODEL,
    title_col: str = "title",
    desc_col: str = "description"
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

    entities_data = []
    triples_data = []

    for i, row in df.iterrows():
        try:
            kg = extract_kg_single(
                row.get(title_col, ""),
                row.get(desc_col, ""),
                model_name
            )

            article_id = i  # DataFrame index를 article_id로 사용
            id_to_name = {}

            # Nodes -> Entities
            for node in kg.get("nodes", []):
                node_id = node.get("id")
                name = node.get("name")
                if node_id and name:
                    id_to_name[node_id] = name
                    entities_data.append({
                        "article_id": article_id,
                        "entity_value": name
                    })

            # Relationships -> Triples
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

        except Exception as e:
            print(f"    Error on row {i}: {e}")
            continue

        if (i + 1) % 10 == 0:
            print(f"    Processed {i + 1}/{len(df)}")

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
    print(f"Nodes: {kg.get('nodes', [])}")
    print(f"Relationships: {kg.get('relationships', [])}")

    unload_model()
