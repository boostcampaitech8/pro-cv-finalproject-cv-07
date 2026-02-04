"""
Google News Crawler 설정 파일
"""
from pathlib import Path

# ============================================================
# 검색 키워드 (Google 검색 문법 지원: OR, AND, 괄호, 따옴표)
# ============================================================

# 카테고리/인덱스 페이지 제외 필터
EXCLUDE_PATTERNS = '-inurl:topic -inurl:section -inurl:category -inurl:tag -inurl:page -inurl:author -inurl:by -inurl:pro'

KEYWORD_QUERIES = {
    "soybean": f'soybean (price OR demand OR supply OR inventory) {EXCLUDE_PATTERNS}',
    # "corn": f'corn (price OR demand OR supply OR inventory) {EXCLUDE_PATTERNS}',
    # "wheat": f'wheat (price OR demand OR supply OR inventory) {EXCLUDE_PATTERNS}',
    # "rice" : f'rice (price OR demand OR supply OR inventory) {EXCLUDE_PATTERNS}',
    # "USDA" : f'united states department of agriculture {EXCLUDE_PATTERNS}',
    # "NASS" : f'national agricultural statistics service {EXCLUDE_PATTERNS}',
    # "soy oil" : f'soy oil (production OR outputs OR supplies OR supply OR biofuel OR biodiesel OR demand OR price) {EXCLUDE_PATTERNS}',
    # "soybean oil" : f'soybean oil (production OR outputs OR supplies OR supply OR biofuel OR biodiesel OR demand OR price) {EXCLUDE_PATTERNS}',
    # "soybean production": f'soybean production {EXCLUDE_PATTERNS}',
    # "sorghum" : f'sorghum (price OR demand OR supply OR inventory) {EXCLUDE_PATTERNS}',
    # "gold":   f'gold (futures OR price OR bullion OR "spot gold" OR XAU) {EXCLUDE_PATTERNS}',
    # "silver": f'silver (futures OR price OR "spot silver") {EXCLUDE_PATTERNS}',
    # "copper": f'copper (futures OR price OR "LME inventory" OR demand OR supply) {EXCLUDE_PATTERNS}',
}

TARGET_MEDIA = [
    "cnbc.com",
    "ft.com",
    "apnews.com",
    "agriculture.com",

    # 차단당한 것들
    # "nytimes.com",
    # "washingtonpost.com", 
    # "farmprogress.com",
    # "world-grain.com",
    # "bloomberg.com",
    # "reuters.com",
    # "wsj.com", 로그인 필요해서 막힘
]

# ============================================================
# 크롤링 설정
# ============================================================
# 수집 기간 (일)
DAYS_BACK = 1

# 키워드당 최대 기사 수
MAX_RESULTS_PER_KEYWORD = 100

# 병렬 처리 워커 수
MAX_WORKERS = 4

# HTTP 요청 타임아웃 (초)
REQUEST_TIMEOUT = 10

# User-Agent
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# ============================================================
# 출력 설정
# ============================================================
OUTPUT_DIR = Path(__file__).parent / "output"

# GNews 설정
GNEWS_LANGUAGE = "en"
GNEWS_COUNTRY = "US"
