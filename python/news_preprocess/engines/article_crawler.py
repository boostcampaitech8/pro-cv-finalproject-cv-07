#!/usr/bin/env python3
"""
뉴스 크롤링 모듈
- Google News에서 뉴스 수집
- URL 디코딩 및 콘텐츠 추출
- google_news_crawler.py 기반
"""

import time
import pandas as pd
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
from gnews import GNews
from googlenewsdecoder import new_decoderv1
from trafilatura import fetch_url, bare_extraction
from trafilatura.settings import use_config
from bs4 import BeautifulSoup
from typing import Optional
import pandas_market_calendars as mcal
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from crawler_config import (
    KEYWORD_QUERIES,
    TARGET_MEDIA,
    OUTPUT_DIR,
    DAYS_BACK,
    MAX_RESULTS_PER_KEYWORD,
    MAX_WORKERS,
    GNEWS_LANGUAGE,
    GNEWS_COUNTRY,
)


# trafilatura 설정 (timeout 단축)
TRAFILATURA_CONFIG = use_config()
TRAFILATURA_CONFIG.set("DEFAULT", "DOWNLOAD_TIMEOUT", "10")

# 도메인별 동시성 제한 (차단 회피)
DOMAIN_SEMAPHORES = {
    "apnews.com": Semaphore(1),
    "cnbc.com": Semaphore(1),
}

# Google 디코딩 동시성 제한 (rate limit 회피)
GOOGLE_DECODE_SEMAPHORE = Semaphore(2)


def get_domain_key(url: str) -> str:
    """도메인 키 생성 (www 제거)"""
    if not url:
        return ""
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc[4:] if netloc.startswith("www.") else netloc
    except Exception:
        return ""


# ============================================================
# URL 처리
# ============================================================
def decode_google_url(google_url: str, delay: float = 0.5) -> str:
    """구글 뉴스 URL -> 원본 URL 디코딩 (rate limit 회피를 위해 호출 간 딜레이)"""
    if not google_url or "news.google.com" not in google_url:
        return google_url
    try:
        with GOOGLE_DECODE_SEMAPHORE:
            time.sleep(delay)
            decoded = new_decoderv1(google_url)
            if decoded and decoded.get('status'):
                return decoded.get('decoded_url', google_url)
    except Exception:
        pass
    return google_url


def normalize_url(url: str) -> str:
    """URL 정규화 (중복 제거용)"""
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        scheme = "https"
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        path = parsed.path.rstrip("/")
        query_params = parse_qs(parsed.query)
        filtered_params = {
            k: v for k, v in query_params.items()
            if not k.startswith(("utm_", "fbclid", "gclid", "ref"))
        }
        query = urlencode(filtered_params, doseq=True) if filtered_params else ""
        return urlunparse((scheme, netloc, path, "", query, ""))
    except Exception:
        return url


# ============================================================
# 콘텐츠 추출
# ============================================================
def extract_article_content(url: str) -> dict:
    """trafilatura로 기사 본문, 저자 추출 + BeautifulSoup으로 og:description 추출"""
    if not url or "news.google.com" in url:
        return {}
    try:
        downloaded = fetch_url(url, config=TRAFILATURA_CONFIG)
        if not downloaded:
            return {}

        # trafilatura로 본문, 저자, 사이트명 추출
        result = bare_extraction(downloaded, include_comments=False)
        text = ""
        authors = ""
        sitename = ""
        if result:
            text = getattr(result, "text", "") or ""
            authors = getattr(result, "author", "") or ""
            sitename = getattr(result, "sitename", "") or ""

        # BeautifulSoup으로 description 추출
        meta_description = ""
        soup = BeautifulSoup(downloaded, 'lxml')

        og = soup.find("meta", attrs={"property": "og:description"})
        if og and og.get("content"):
            meta_description = og["content"].strip()

        if not meta_description:
            meta = soup.find("meta", attrs={"name": "description"})
            if meta and meta.get("content"):
                meta_description = meta["content"].strip()

        return {
            "text": text,
            "authors": authors,
            "meta_description": meta_description,
            "sitename": sitename,
        }
    except Exception:
        pass
    return {}


def format_newsapi_style(full_text: str, preview_length: int = 200) -> str:
    """본문을 NewsAPI 스타일로 포맷: '앞부분... [+남은글자수 chars]'"""
    if not full_text:
        return ""

    if len(full_text) <= preview_length:
        return full_text

    preview = full_text[:preview_length].rstrip()
    remaining = len(full_text) - preview_length
    return f"{preview}... [+{remaining} chars]"


# ============================================================
# 기사 처리
# ============================================================
def process_article(art: dict, keyword_label: str, collect_date_str: Optional[str] = None) -> dict:
    """단일 기사 처리: URL 디코딩 + trafilatura로 콘텐츠 추출"""
    google_url = art.get('url', '')
    raw_title = art.get('title', '')

    doc_url = decode_google_url(google_url)

    content = {}
    if doc_url and doc_url != google_url:
        domain_key = get_domain_key(doc_url)
        sem = DOMAIN_SEMAPHORES.get(domain_key)
        if sem:
            with sem:
                content = extract_article_content(doc_url)
        else:
            content = extract_article_content(doc_url)

    full_text = content.get("text", "")
    all_text = format_newsapi_style(full_text, preview_length=200)

    meta_desc = content.get("meta_description", "")
    gnews_desc = (
        art.get("description")
        or art.get("desc")
        or art.get("snippet")
        or ""
    )
    title_clean = raw_title.rsplit(' - ', 1)[0].strip() if ' - ' in raw_title else raw_title.strip()

    # GNews description에서 publisher name 제거 (title + publisher 형태 처리)
    publisher = art.get('publisher', {})
    publisher_name = publisher.get('title', '') if isinstance(publisher, dict) else str(publisher)
    if gnews_desc and publisher_name:
        gnews_desc_stripped = str(gnews_desc).strip()
        # 끝에 publisher name이 붙어있으면 제거
        if gnews_desc_stripped.endswith(publisher_name):
            gnews_desc = gnews_desc_stripped[:-len(publisher_name)].strip()
        # " - Publisher" 형태도 처리
        if gnews_desc_stripped.endswith(f" - {publisher_name}"):
            gnews_desc = gnews_desc_stripped[:-(len(publisher_name) + 3)].strip()
    content_empty = not any([
        content.get("text"),
        content.get("meta_description"),
        content.get("authors"),
        content.get("sitename"),
    ])

    if meta_desc and meta_desc.strip() != title_clean:
        desc = meta_desc
    elif full_text:
        desc = full_text[:200].strip() if len(full_text) > 200 else full_text
    elif content_empty and gnews_desc and str(gnews_desc).strip() != title_clean:
        desc = str(gnews_desc).strip()
    else:
        desc = ""

    # description 검증: title과 같거나 20자 미만이면 all_text로 대체
    desc_normalized = desc.lower().strip() if desc else ""
    title_normalized = title_clean.lower().strip() if title_clean else ""
    if desc_normalized == title_normalized or len(desc.strip()) < 20:
        # all_text에서 preview 부분만 추출 (... [+N chars] 제거)
        if full_text and len(full_text.strip()) >= 20:
            desc = full_text[:200].strip() if len(full_text) > 200 else full_text
        else:
            desc = ""  # all_text도 없거나 짧으면 빈 문자열 (deduplicate에서 제거됨)

    authors = content.get("authors", "")
    title = raw_title.rsplit(' - ', 1)[0] if ' - ' in raw_title else raw_title

    meta_site_name = content.get("sitename", "")
    if not meta_site_name:
        publisher = art.get('publisher', {})
        meta_site_name = publisher.get('title', '') if isinstance(publisher, dict) else str(publisher)

    return {
        "title": title,
        "doc_url": doc_url,
        "description": desc,
        "all_text": all_text,
        "authors": authors,
        "publish_date": art.get('published date', ''),
        "collect_date": collect_date_str,
        "meta_site_name": meta_site_name,
        "key_word": keyword_label,
    }


# ============================================================
# 뉴스 수집
# ============================================================
def fetch_news(query: str, keyword_label: str, start_date: tuple, end_date: tuple,
               max_results: int = 100, filter_dt: Optional[datetime] = None,
               filter_end_dt: Optional[datetime] = None,
               collect_date_str: Optional[str] = None) -> list[dict]:
    """Google News에서 뉴스 수집 (날짜 필터링 후 병렬 처리)"""
    print(f"  Fetching: '{keyword_label}'")

    gn = GNews(
        language=GNEWS_LANGUAGE,
        country=GNEWS_COUNTRY,
        max_results=max_results,
        start_date=start_date,
        end_date=end_date
    )

    try:
        articles = gn.get_news(query)
    except Exception as e:
        print(f"  Error fetching news: {e}")
        return []

    if not articles:
        print(f"  No articles found for '{keyword_label}'")
        return []

    print(f"  Found {len(articles)} raw articles")

    # 날짜 필터링 먼저 수행 (세부 처리 전)
    if filter_dt or filter_end_dt:
        filtered_articles = []
        for art in articles:
            pub_str = art.get('published date', '')
            try:
                pub_dt = pd.to_datetime(pub_str, utc=True, errors='coerce')
                if pd.isna(pub_dt):
                    continue
                if filter_dt and pub_dt < filter_dt:
                    continue
                if filter_end_dt and pub_dt >= filter_end_dt:
                    continue
                filtered_articles.append(art)
            except:
                pass
        filter_msg = f">= {filter_dt.strftime('%Y-%m-%d %H:%M')}" if filter_dt else ""
        if filter_end_dt:
            filter_msg += f", < {filter_end_dt.strftime('%Y-%m-%d %H:%M')}" if filter_msg else f"< {filter_end_dt.strftime('%Y-%m-%d %H:%M')}"
        print(f"  After date filter ({filter_msg}): {len(filtered_articles)} articles")
        articles = filtered_articles

    if not articles:
        print(f"  No articles after date filter for '{keyword_label}'")
        return []

    print(f"  Processing {len(articles)} articles in parallel...")

    rows = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_article, art, keyword_label, collect_date_str): i
            for i, art in enumerate(articles)
        }

        for future in as_completed(futures):
            try:
                result = future.result()
                rows.append(result)
            except Exception as e:
                print(f"    Error processing article: {e}")

    print(f"  Completed: {len(rows)} articles for '{keyword_label}'")
    return rows


def adjust_collect_date_to_trading_day(df: pd.DataFrame, exchange: str = "NYSE") -> pd.DataFrame:
    """collect_date가 휴장일/주말이면 직전 개장일로 변경"""
    if df.empty or 'collect_date' not in df.columns:
        return df

    nyse = mcal.get_calendar(exchange)
    unique_dates = df['collect_date'].dropna().unique()

    if len(unique_dates) == 0:
        return df

    # 조회 범위: 최소 날짜 30일 전 ~ 최대 날짜 (직전 개장일 찾기 위한 여유)
    min_date = pd.Timestamp(min(unique_dates)) - pd.Timedelta(days=30)
    max_date = pd.Timestamp(max(unique_dates))
    trading_days = set(nyse.valid_days(start_date=min_date, end_date=max_date).date)

    def find_prev_trading_day(d):
        if d is None or pd.isna(d):
            return d
        current = d
        while current not in trading_days:
            current -= timedelta(days=1)
        return current

    adjusted_count = 0
    new_dates = []
    for d in df['collect_date']:
        new_d = find_prev_trading_day(d)
        if new_d != d:
            adjusted_count += 1
        new_dates.append(new_d)

    df['collect_date'] = new_dates
    print(f"  Holiday adjustment: {adjusted_count} articles' collect_date moved to previous trading day")
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """URL 정규화 + Title 기반 중복 제거 + 콘텐츠 없는 기사 제외"""
    if df.empty:
        return df

    original_count = len(df)

    # description과 all_text 둘 다 비어있는 기사 제외
    df = df[~((df["description"].isna() | (df["description"] == "")) &
              (df["all_text"].isna() | (df["all_text"] == "")))]
    after_content_filter = len(df)

    print(df.head(5))

    # URL 중복 제거
    df["_normalized_url"] = df["doc_url"].apply(normalize_url)
    df["_normalized_title"] = df["title"].str.lower().str.replace(r'[^\w\s]', '', regex=True).str.strip()

    df = df.drop_duplicates(subset=["_normalized_url"], keep="first")
    df = df.drop_duplicates(subset=["_normalized_title"], keep="first")
    df = df.drop(columns=["_normalized_url", "_normalized_title"])

    print(df.head(5))
    print(f"  Content filter: {original_count} -> {after_content_filter} (removed {original_count - after_content_filter} empty)")
    print(f"  Deduplication: {after_content_filter} -> {len(df)} articles")
    return df.reset_index(drop=True)


# ============================================================
# 메인 크롤링 함수
# ============================================================
def crawl_article(days_back: Optional[int] = None,
               keywords: Optional[dict] = None,
               target_media: Optional[list] = None,
               max_results: Optional[int] = None,
               start_hour: int = 0,
               start_minute: int = 0,
               end_date_str: Optional[str] = None) -> pd.DataFrame:
    """
    뉴스 크롤링 메인 함수

    Args:
        days_back: 수집 기간 (일), None이면 config 값 사용
        keywords: 키워드 쿼리 dict, None이면 config 값 사용
        target_media: 타겟 미디어 리스트, None이면 config 값 사용
        max_results: 키워드당 최대 결과 수, None이면 config 값 사용
        start_hour: 필터링 시작 시간 (0-23)
        start_minute: 필터링 시작 분 (0-59)
        end_date_str: 수집 종료 날짜 (YYYY-MM-DD 형식), None이면 현재 시각 사용.
                      지정 시 해당 날짜의 start_hour:start_minute 미만(<)으로 필터링

    Returns:
        크롤링된 뉴스 DataFrame
    """
    days_back = days_back or DAYS_BACK
    keywords = keywords or KEYWORD_QUERIES
    target_media = target_media if target_media is not None else TARGET_MEDIA
    max_results = max_results or MAX_RESULTS_PER_KEYWORD

    print("=" * 60)
    print("News Crawler")
    print("=" * 60)

    # 날짜 범위 설정
    UTC = timezone.utc

    if end_date_str:
        end_base = datetime.strptime(end_date_str, '%Y-%m-%d').replace(tzinfo=UTC)
        end_utc = end_base.replace(
            hour=start_hour, minute=start_minute, second=0, microsecond=0
        )
    else:
        end_utc = datetime.now(UTC)

    # 시작 시간: days_back일 전 + 지정된 시:분
    start_utc = (end_utc - timedelta(days=days_back)).replace(
        hour=start_hour, minute=start_minute, second=0, microsecond=0
    )

    start_date = (start_utc.year, start_utc.month, start_utc.day)
    end_date = (end_utc.year, end_utc.month, end_utc.day)

    # collect_date 문자열 (필터 시작 날짜)
    collect_date_str = start_utc.strftime('%Y-%m-%d')

    print(f"Date filter: >= {start_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC, < {end_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"Keywords: {list(keywords.keys())}")
    print(f"Target media: {len(target_media)} sites")
    print()

    all_rows = []

    # 검색 작업 목록 생성
    search_tasks = []

    # 1) 전체 검색 작업 (먼저 수행)
    for keyword, query in keywords.items():
        search_tasks.append((query, keyword, keyword))

    # 2) TARGET_MEDIA 사이트별 + 키워드별 검색 작업
    if target_media:
        for keyword, query in keywords.items():
            for site in target_media:
                site_query = f"site:{site} {query}"
                search_tasks.append((site_query, keyword, f"{keyword}@{site}"))

    print(f"Total search tasks: {len(search_tasks)}")
    print()

    # 병렬 검색 실행
    def run_search(task):
        query, keyword_label, display_label = task
        return display_label, fetch_news(
            query=query,
            keyword_label=keyword_label,
            start_date=start_date,
            end_date=end_date,
            max_results=max_results,
            filter_dt=start_utc,
            filter_end_dt=end_utc,
            collect_date_str=collect_date_str
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(run_search, task): task for task in search_tasks}
        for future in as_completed(futures):
            try:
                label, rows = future.result()
                print(f"  [{label}] -> {len(rows)} articles")
                all_rows.extend(rows)
            except Exception as e:
                print(f"  Error: {e}")

    # DataFrame 생성
    df = pd.DataFrame(all_rows)
    print(f"\nTotal articles collected: {len(df)}")

    if df.empty:
        print("No articles found.")
        return df

    # UTC 기준으로 수집 후 필터링 (start_utc 이상, end_utc 미만)
    df["_published_dt"] = pd.to_datetime(df["publish_date"], utc=True, errors="coerce")
    before_time_filter = len(df)
    dt_filter = df["_published_dt"].notna() & (df["_published_dt"] >= start_utc) & (df["_published_dt"] < end_utc)
    df = df[dt_filter]
    print(f"Post-filter by published date (UTC): {before_time_filter} -> {len(df)} articles")
    df = df.drop(columns=["_published_dt"])

    # 데이터 타입 변환
    # publish_date: datetime (시간 포함)
    df['publish_date'] = pd.to_datetime(df['publish_date'], utc=True, errors='coerce')
    # collect_date: date
    df['collect_date'] = pd.to_datetime(df['collect_date'], errors='coerce').dt.date

    # 휴장일 처리: collect_date가 휴장일/주말이면 직전 개장일로 변경
    df = adjust_collect_date_to_trading_day(df)

    # 중복 제거
    print("\nRemoving duplicates...")
    df = deduplicate(df)

    # 컬럼 정렬
    columns = ["title", "doc_url", "description", "all_text", "authors", "publish_date", "collect_date", "meta_site_name", "key_word"]
    df = df[columns]

    print(f"Final article count: {len(df)}")
    return df


def save_to_csv(df: pd.DataFrame, output_dir=None) -> str:
    """DataFrame을 CSV로 저장"""
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"google_news_{timestamp}.csv"

    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Saved to: {output_file}")

    return str(output_file)


if __name__ == "__main__":
    df = crawl_article(
        days_back=1,
        start_hour=13,
        start_minute=45,
    )
    if not df.empty:
        save_to_csv(df)
        print("\nArticles per keyword:")
        print(df["key_word"].value_counts().to_string())
