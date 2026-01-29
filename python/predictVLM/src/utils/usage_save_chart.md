# VLM Chart Image Generator

VLM(Vision Language Model) 실험을 위한 캔들스틱 차트 이미지 생성 도구입니다.

## Requirements

```bash
pip install pandas mplfinance matplotlib pillow
```

## Usage

### 기본 실행

```bash
python save_chart_image.py
```

### 커스텀 옵션

```bash
python save_chart_image.py \
    --data-path ../../data/corn_future_price.csv \
    --output-dir ./images \
    --spans 5 10 20 \
    --end-start 2025-10-29 \
    --end-stop 2024-05-29 \
    --windows 20 \
    --emas 0 \
    --chart-type candle \
    --image-size 448
```

### 여러 윈도우/EMA 조합 생성

```bash
# window 5, 20, 60 + EMA 없음, EMA 20 조합 = 6가지 폴더 생성
python save_chart_image.py \
    --windows 5 20 60 \
    --emas 0 20
```

## Parameters

| 인자 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--data-path` | str | `../../data/corn_future_price.csv` | 가격 데이터 CSV 파일 경로 |
| `--spans` | int[] | `5 10 20` | 계산할 EMA span 목록 |
| `--end-start` | str | `2025-10-29` | 생성할 날짜 범위의 종료일 |
| `--end-stop` | str | `2024-05-29` | 생성할 날짜 범위의 시작일 |
| `--windows` | int[] | `20` | 차트 윈도우 크기 목록 |
| `--emas` | int[] | `0` | 차트에 표시할 EMA (0=표시 안함) |
| `--output-dir` | str | `images` | 이미지 저장 기본 디렉토리 |
| `--chart-type` | str | `candle` | 차트 유형 (`candle` 또는 `ohlc`) |
| `--image-size` | int | `448` | 출력 이미지 크기 (정사각형) |

## Output Structure

```
images/
├── window_5_ema0/
│   ├── 2024-05-29.png
│   ├── 2024-05-30.png
│   └── ...
├── window_20_ema0/
│   ├── 2024-05-29.png
│   └── ...
└── window_20_ema20/
    ├── 2024-05-29.png
    └── ...
```

## Data Format

입력 CSV 파일은 다음 컬럼을 포함해야 합니다:

| 컬럼 | 설명 |
|------|------|
| `time` (index) | 날짜 (YYYY-MM-DD) |
| `open` | 시가 |
| `high` | 고가 |
| `low` | 저가 |
| `close` | 종가 |
| `Volume` 또는 `volume` | 거래량 |

## Image Generation Logic

- 파일명 `2024-11-26.png`는 **2024-11-26을 예측하기 위한** 차트입니다
- 차트에 포함된 데이터: `t-window ~ t-1` (예측 대상일 미포함)
- 예: window=20, 예측일=2024-11-26 → 2024-10-28 ~ 2024-11-25 데이터로 차트 생성

## Examples

```bash
# OHLC 차트로 생성
python save_chart_image.py --chart-type ohlc

# 512x512 크기로 생성
python save_chart_image.py --image-size 512

# 특정 기간만 생성
python save_chart_image.py --end-start 2025-01-31 --end-stop 2025-01-01

# EMA 20 포함 차트 생성
python save_chart_image.py --emas 20 --spans 20

# 여러 EMA span 계산 (5, 10, 20, 50, 100)
python save_chart_image.py --spans 5 10 20 50 100

# 여러 윈도우 크기로 이미지 생성 (5일, 20일, 60일)
python save_chart_image.py --windows 5 20 60

# 여러 EMA를 차트에 표시 (EMA 없음 + EMA 10 + EMA 20)
python save_chart_image.py --emas 0 10 20 --spans 10 20

# 윈도우 + EMA 조합 (6개 폴더 생성: 3 windows × 2 emas)
python save_chart_image.py --windows 5 20 60 --emas 0 20 --spans 20
```

## Notes

- `--spans`는 **계산할** EMA 목록 (DataFrame에 컬럼 추가)
- `--emas`는 **차트에 표시할** EMA 목록 (0은 EMA 미표시)
- `--emas`에 지정한 값은 반드시 `--spans`에 포함되어야 함
- 윈도우와 EMA 조합마다 별도 폴더 생성: `window_{w}_ema{ema}/`
