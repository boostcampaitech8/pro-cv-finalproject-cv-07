"""
Report Generation Prompts
- GPT API에 전달할 시스템/유저 프롬프트 정의
"""

REPORT_SYSTEM_PROMPT = (
    "You are a senior commodity market analyst. "
    "You will receive predicted price data and prediction chart images for six commodities "
    "(gold, silver, copper, corn, wheat, soybean). "
    "Write a comprehensive daily market report in **Korean** using Markdown format. "
    "Include trend analysis, cross-commodity insights, and key risk factors."
)


def build_report_prompt(price_data_text: str) -> str:
    """가격 데이터를 삽입한 유저 프롬프트 반환."""
    return (
        "아래는 오늘 기준 향후 20일간 6개 원자재(금, 은, 구리, 옥수수, 밀, 대두)의 예측 가격 데이터입니다.\n\n"
        f"{price_data_text}\n\n"
        "첨부된 이미지는 DeepAR 및 TFT 모델의 예측 차트(각각 w5, w20, w60 window)입니다.\n\n"
        "다음 항목을 포함하는 마크다운 리포트를 작성해 주세요:\n"
        "1. 전체 시장 요약\n"
        "2. 품목별 가격 전망 (금, 은, 구리, 옥수수, 밀, 대두)\n"
        "3. 모델 간 예측 비교 (DeepAR vs TFT)\n"
        "4. 주요 리스크 요인\n"
        "5. 투자 시사점\n"
    )
