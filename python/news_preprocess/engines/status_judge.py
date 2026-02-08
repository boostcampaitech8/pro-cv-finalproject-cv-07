#!/usr/bin/env python3
"""
뉴스 T/F 판단 모듈
- Mistral API를 사용하여 뉴스가 상품 선물 가격 예측에 관련있는지 판단
"""

import os
import pandas as pd
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 기본 모델
DEFAULT_MODEL = "open-mistral-nemo-2407"

# 프롬프트 템플릿
SYSTEM_PROMPT = """You are a Commodities Trader looking for market signals.
Your task is to determine if the news article contains **any information that could influence the futures price** (up or down) of the target assets.

**Target Assets:**
1. **Agriculture:** Grains (Corn, Wheat, Soybean)
2. **Metals:** Gold, Silver, Copper

**Classification Criteria:**

**RELEVANT (T): Potential Price Mover**
* Mark 'T' if the content suggests any factor that could lead to a price change (Bullish or Bearish).
* This includes:
    * Supply/Demand changes (production, weather, trade).
    * Market Sentiment (fears, hype, geopolitical tension).
    * Related economic indicators (inflation, dollar strength) that usually move commodities.
    * **Broad Logic:** If a trader might care about this news for *any* reason, mark it 'T'.

**NOT RELEVANT (F): Completely Unrelated**
* Mark 'F' ONLY if the news has **ZERO** connection to the commodity markets (e.g., Sports, Celebrity gossip, totally unrelated tech news).

**Output:** ONLY 'T' or 'F'."""

USER_TEMPLATE = """Title: {title}
Description: {description}
Keyword: {key_word}

Is this article relevant to agricultural commodities market news? Answer T or F only."""


# ============================================================
# Mistral API 클라이언트
# ============================================================
_client = None


def _get_client():
    """Mistral API 클라이언트 (OpenAI SDK 호환, 싱글톤)"""
    global _client
    if _client is None:
        from openai import OpenAI
        api_key = os.environ.get("MISTRAL_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY not set. Put it in .env or environment.")
        _client = OpenAI(api_key=api_key, base_url="https://api.mistral.ai/v1")
    return _client


def load_model(model_name: str = DEFAULT_MODEL):
    """호환성 유지용 (API 모드에서는 클라이언트 반환)"""
    return _get_client(), None


def unload_model():
    """클라이언트 해제"""
    global _client
    _client = None
    print("Mistral API client released")


# ============================================================
# T/F 판단 함수
# ============================================================
def _build_messages(title: str, description: str, key_word: str) -> list:
    """Chat 형식 메시지 생성"""
    user_content = USER_TEMPLATE.format(
        title=str(title)[:200],
        description=str(description)[:300] if pd.notna(description) else "",
        key_word=str(key_word) if pd.notna(key_word) else ""
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]


def _parse_response(response: str) -> str:
    """응답에서 T/F 추출"""
    clean_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    clean_response = re.sub(r'<think>.*', '', clean_response, flags=re.DOTALL).strip()
    clean_response = clean_response.upper()

    if 'T' in clean_response.split()[-5:]:
        return 'T'
    if 'F' in clean_response.split()[-5:]:
        return 'F'

    if re.search(r'\bT\b', clean_response) or clean_response.startswith('T'):
        return 'T'
    if re.search(r'\bF\b', clean_response) or clean_response.startswith('F'):
        return 'F'
    return 'F'


def judge_single(title: str, description: str, key_word: str,
                  model_name: str = DEFAULT_MODEL, client=None) -> str:
    """
    단일 기사 T/F 판단 (Mistral API)

    Args:
        title: 기사 제목
        description: 기사 설명
        key_word: 검색 키워드
        model_name: Mistral 모델명
        client: OpenAI 호환 클라이언트 (None이면 싱글톤 사용)

    Returns:
        'T' (관련 있음) 또는 'F' (관련 없음)
    """
    if client is None:
        client = _get_client()
    messages = _build_messages(title, description, key_word)

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=20,
        temperature=0.1,
    )

    text = response.choices[0].message.content
    return _parse_response(text)


def judge_batch(df: pd.DataFrame, model_name: str = DEFAULT_MODEL,
                title_col: str = "title", desc_col: str = "description",
                keyword_col: str = "key_word") -> list[str]:
    """
    배치 T/F 판단 (병렬 4 workers)

    Args:
        df: DataFrame (title, description, key_word 컬럼 필요)
        model_name: 사용할 모델명
        title_col: 제목 컬럼명
        desc_col: 설명 컬럼명
        keyword_col: 키워드 컬럼명

    Returns:
        T/F 결과 리스트
    """
    print(f"Judging {len(df)} articles (4 workers)...")

    client = _get_client()
    predictions = ['F'] * len(df)

    def _judge_one(idx_row):
        idx, row = idx_row
        try:
            pred = judge_single(
                row[title_col],
                row.get(desc_col, ""),
                row.get(keyword_col, ""),
                model_name,
                client=client,
            )
            return idx, pred
        except Exception as e:
            print(f"  Error on row {idx}: {e}")
            return idx, 'F'

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_judge_one, (i, row)): i
            for i, row in df.iterrows()
        }
        for future in tqdm(as_completed(futures), total=len(df), desc="T/F Classification"):
            idx, pred = future.result()
            predictions[idx] = pred

    return predictions


def filter_relevant(df: pd.DataFrame, model_name: str = DEFAULT_MODEL,
                   title_col: str = "title", desc_col: str = "description",
                   keyword_col: str = "key_word") -> pd.DataFrame:
    """
    관련 있는 기사만 필터링 (T인 것만 반환)

    Args:
        df: DataFrame
        model_name: 사용할 모델명

    Returns:
        T로 판단된 기사만 포함된 DataFrame
    """
    predictions = judge_batch(df, model_name, title_col, desc_col, keyword_col)
    df = df.copy()
    df['filter_status'] = predictions

    relevant = df[df['filter_status'] == 'T'].copy()
    print(f"Filtered: {len(df)} -> {len(relevant)} relevant articles ({len(relevant)/len(df)*100:.1f}%)")

    return relevant


if __name__ == "__main__":
    import sys
    _NEWS_PREPROCESS_ROOT = Path(__file__).resolve().parent.parent
    _REPO_ROOT = _NEWS_PREPROCESS_ROOT.parent.parent
    from dotenv import load_dotenv
    load_dotenv(_REPO_ROOT / ".env")

    test_title = "Soybean prices surge as drought hits Argentina"
    test_desc = "Argentine farmers face significant crop losses due to prolonged dry conditions"
    test_keyword = "soybean"

    result = judge_single(test_title, test_desc, test_keyword)
    print(f"Test result: {result}")

    unload_model()