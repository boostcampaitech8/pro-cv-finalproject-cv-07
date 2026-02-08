#!/usr/bin/env python3
"""
뉴스 T/F 판단 모듈
- LLM을 사용하여 뉴스가 상품 선물 가격 예측에 관련있는지 판단
- model_benchmark_transformers.py 기반
"""

import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import gc
from pathlib import Path


# ============================================================
# 설정
# ============================================================
MODEL_CACHE_DIR = Path("/data/ephemeral/home/tena/model_cache")

# HuggingFace 캐시 디렉토리 설정
os.environ["HF_HOME"] = str(MODEL_CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(MODEL_CACHE_DIR / "transformers")
os.environ["HF_HUB_CACHE"] = str(MODEL_CACHE_DIR / "hub")

# 기본 모델
DEFAULT_MODEL = "mistralai/Mistral-Nemo-Instruct-2407"

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
# 모델 관리 (싱글톤)
# ============================================================
_model = None
_tokenizer = None
_current_model_name = None


def load_model(model_name: str = DEFAULT_MODEL):
    """모델 로드 (싱글톤 패턴)"""
    global _model, _tokenizer, _current_model_name

    if _model is not None and _current_model_name == model_name:
        return _model, _tokenizer

    # 기존 모델 정리
    if _model is not None:
        del _model, _tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    print(f"Loading model: {model_name}...")
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    _tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=str(MODEL_CACHE_DIR / "hub"),
    )

    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    _model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=str(MODEL_CACHE_DIR / "hub"),
    )
    _model.eval()
    _current_model_name = model_name

    print(f"Model loaded: {model_name}")
    return _model, _tokenizer


def unload_model():
    """모델 메모리 해제"""
    global _model, _tokenizer, _current_model_name

    # 변수가 정의되지 않았거나 None인 경우 처리
    if '_model' not in globals() or _model is None:
        return

    del _model, _tokenizer
    _model = None
    _tokenizer = None
    _current_model_name = None
    gc.collect()
    torch.cuda.empty_cache()
    print("Model unloaded")


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


def judge_single(title: str, description: str, key_word: str, model_name: str = DEFAULT_MODEL) -> str:
    """
    단일 기사 T/F 판단

    Args:
        title: 기사 제목
        description: 기사 설명
        key_word: 검색 키워드

    Returns:
        'T' (관련 있음) 또는 'F' (관련 없음)
    """
    model, tokenizer = load_model(model_name)
    messages = _build_messages(title, description, key_word)

    if hasattr(tokenizer, 'apply_chat_template'):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = f"System: {SYSTEM_PROMPT}\n\nUser: {messages[1]['content']}\n\nAssistant:"

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return _parse_response(response)


def judge_batch(df: pd.DataFrame, model_name: str = DEFAULT_MODEL,
                title_col: str = "title", desc_col: str = "description",
                keyword_col: str = "key_word") -> list[str]:
    """
    배치 T/F 판단

    Args:
        df: DataFrame (title, description, key_word 컬럼 필요)
        model_name: 사용할 모델명
        title_col: 제목 컬럼명
        desc_col: 설명 컬럼명
        keyword_col: 키워드 컬럼명

    Returns:
        T/F 결과 리스트
    """
    predictions = []

    print(f"Judging {len(df)} articles...")

    for i, row in df.iterrows():
        try:
            pred = judge_single(
                row[title_col],
                row.get(desc_col, ""),
                row.get(keyword_col, ""),
                model_name
            )
            predictions.append(pred)
        except Exception as e:
            print(f"  Error on row {i}: {e}")
            predictions.append('F')

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(df)}")

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
    # 테스트
    test_title = "Soybean prices surge as drought hits Argentina"
    test_desc = "Argentine farmers face significant crop losses due to prolonged dry conditions"
    test_keyword = "soybean"

    result = judge_single(test_title, test_desc, test_keyword)
    print(f"Test result: {result}")

    unload_model()