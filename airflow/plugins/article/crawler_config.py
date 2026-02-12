"""
Google News Crawler 설정 파일
"""
from pathlib import Path

# 카테고리/인덱스 페이지 제외 필터
#EXCLUDE_PATTERNS = '-inurl:topic -inurl:section -inurl:category -inurl:tag -inurl:page -inurl:author -inurl:by -inurl:pro'

KEYWORD_QUERIES = {
    "soybean": f'soybean (price OR demand OR supply OR inventory OR production)',
    "corn": f'corn (price OR demand OR supply OR inventory)',
    "wheat": f'wheat (price OR demand OR supply OR inventory)',
    "rice" : f'rice (price OR demand OR supply OR inventory)',
    "USDA" : f'united states department of agriculture',
    "NASS" : f'national agricultural statistics service',
    "soy oil" : f'soy oil (production OR outputs OR supplies OR supply OR biofuel OR biodiesel OR demand OR price)',
    "soybean oil" : f'soybean oil (production OR outputs OR supplies OR supply OR biofuel OR biodiesel OR demand OR price)',
    "sorghum" : f'sorghum (price OR demand OR supply OR inventory)', # 이건 다 soybean으로 매칭
    "gold":   f'gold (futures OR price OR bullion OR "spot gold" OR XAU)',
    "silver": f'silver (futures OR price OR "spot silver")',
    "copper": f'copper (futures OR price OR "LME inventory" OR demand OR supply)',
}

# key_word 저장 시 매핑 (세부 키워드 → 대표 카테고리)
KEYWORD_MAPPING = {
    "USDA": "crops-general",
    "NASS": "crops-general",
    "sorghum": "crops-general",
    "soy oil": "soybean",
    "soybean oil": "soybean",
    "soybean production": "soybean",
}

TARGET_MEDIA = [
    # "cnbc.com",
    # "ft.com",
    # "apnews.com",
    # "agriculture.com",

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
