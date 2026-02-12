#!/usr/bin/env python3
"""
날짜 범위 일괄 수집 스크립트
Usage:
  python batch_collect.py --start 2025-11-10 --end 2026-02-07
  python batch_collect.py --start 2025-11-10 --end 2026-02-07 --skip-embedding --skip-vectordb

crawler_config.py 에서 키워드 확인
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# 경로 설정
ARTICLE_DIR = Path(__file__).resolve().parent        # plugins/article/
AIRFLOW_ROOT = ARTICLE_DIR.parent.parent             # airflow/
REPO_ROOT = AIRFLOW_ROOT.parent                      # pro-cv-finalproject-cv-07/

sys.path.insert(0, str(ARTICLE_DIR))
sys.path.insert(0, str(AIRFLOW_ROOT / 'connections'))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from main_article import run_pipeline
from status_judge import unload_model


def date_range(start_str: str, end_str: str):
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates


def main():
    parser = argparse.ArgumentParser(description="날짜 범위 일괄 뉴스 수집")
    parser.add_argument("--start", type=str, required=True, help="시작 날짜 (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="종료 날짜 (YYYY-MM-DD)")
    parser.add_argument("--days-back", type=int, default=1, help="날짜당 수집 기간 (일)")
    parser.add_argument("--start-hour", type=int, default=13)
    parser.add_argument("--start-minute", type=int, default=45)
    parser.add_argument("--sleep", type=int, default=5, help="날짜 간 대기 시간 (초)")
    parser.add_argument("--skip-tf-filter", action="store_true")
    parser.add_argument("--skip-embedding", action="store_true")
    parser.add_argument("--skip-features", action="store_true")
    parser.add_argument("--skip-vectordb", action="store_true")
    parser.add_argument("--skip-graphdb", action="store_true")
    parser.add_argument("--skip-bigquery", action="store_true")
    parser.add_argument("--save-csv", action="store_true", default=True)
    parser.add_argument("--no-save-csv", action="store_false", dest="save_csv")
    parser.add_argument("--model-name", type=str, default=None)
    args = parser.parse_args()

    dates = date_range(args.start, args.end)
    total = len(dates)

    print("=" * 60)
    print(f"Batch Collection: {args.start} ~ {args.end}")
    print(f"Total dates: {total}")
    print(f"Sleep between dates: {args.sleep}s")
    print("=" * 60)

    success = 0
    failed = []

    for i, end_date in enumerate(dates, 1):
        print(f"\n{'#' * 60}")
        print(f"# [{i}/{total}] Collecting: {end_date}")
        print(f"{'#' * 60}")

        try:
            df = run_pipeline(
                days_back=args.days_back,
                start_hour=args.start_hour,
                start_minute=args.start_minute,
                end_date_str=end_date,
                skip_tf_filter=args.skip_tf_filter,
                skip_embedding=args.skip_embedding,
                skip_features=args.skip_features,
                skip_vectordb=args.skip_vectordb,
                skip_graphdb=args.skip_graphdb,
                skip_bigquery=args.skip_bigquery,
                save_csv=args.save_csv,
                model_name=args.model_name,
                keep_model_loaded=True,
            )
            count = len(df) if df is not None and not df.empty else 0
            print(f"\n  => {end_date}: {count} articles collected")
            success += 1
        except Exception as e:
            print(f"\n  => {end_date}: FAILED - {e}")
            failed.append(end_date)

        if i < total:
            print(f"\n  Waiting {args.sleep}s ...")
            time.sleep(args.sleep)

    # 모델 최종 해제
    try:
        unload_model()
    except Exception:
        pass

    # 결과 요약
    print("\n" + "=" * 60)
    print("Batch Collection Complete")
    print("=" * 60)
    print(f"Success: {success}/{total}")
    if failed:
        print(f"Failed dates: {failed}")


if __name__ == "__main__":
    main()