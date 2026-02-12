from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from app.models import Product, News, DailyPrice, Favorite, PredictPrice, Report
from datetime import datetime, date


async def get_product_detail(db: Session, product_id: int, user_id: int = None):
    if not user_id:
        return None

    product = db.query(Product).filter(Product.product_id == product_id).first()
    if not product:
        return None

    emoji_map = {
        "ZC=F": "🌽", "ZW=F": "🌾", "ZS=F": "🫘",
        "GC=F": "💰", "SI=F": "🪙", "HG=F": "🧱"
    }
    emoji = emoji_map.get(product.ticker, "📦")

    latest_price = (
        db.query(DailyPrice)
        .filter(DailyPrice.product_id == product.product_id)
        .order_by(desc(DailyPrice.base_date))
        .first()
    )
    
    news_items = (
        db.query(News)
        .filter(News.product_id == product.product_id)
        .order_by(desc(News.published_at))
        .limit(3)
        .all()
    )
    
    is_favorite = False
    fav_exists = db.query(Favorite).filter(
        Favorite.member_id == user_id,
        Favorite.product_id == product_id
    ).first()
    if fav_exists:
        is_favorite = True
    
    is_positive = False
    if latest_price and latest_price.price_change > 0:
        is_positive = True
    
    sign = "+" if is_positive else ""

    return {
        "id": product.product_id,
        "name_ko": product.product_name,
        "ticker": product.ticker,
        "emoji": emoji,
        "price": f"${latest_price.closing_price:,.2f}" if latest_price else "$0.00",
        "change_display": f"{sign}{latest_price.price_change:,.3f} ({latest_price.change_rate:+.2f}%)",
        "is_positive": is_positive,
        "is_favorite": is_favorite,
        "news_list": [
            {
                "title": n.title,
                "url": n.news_url,
                "site": n.site_name,
                "time": n.published_at.strftime("%Y-%m-%d %H:%M")
            } for n in news_items
        ]
    }
    
    
async def toggle_favorite_status(db: Session, member_id: int, product_id: int, is_favorite: bool):
    try:
        existing_fav = db.query(Favorite).filter_by(member_id=member_id, product_id=product_id).first()

        if is_favorite:
            if not existing_fav:
                new_item = Favorite(member_id=member_id, product_id=product_id)
                db.add(new_item)
                db.flush() 
        else:
            if existing_fav:
                db.delete(existing_fav)
                db.flush()

        db.commit() 
        return True
    except Exception as e:
        db.rollback()
        print(f"Cloud SQL 저장 오류: {e}")
        return False
    

async def get_report_by_ticker(report_db: Session, ticker: str):
    today = date.today()
    report = report_db.query(Report).filter(
        Report.keyword == ticker,
        Report.publish_date == today
    ).first()

    if not report:
        return None

    return report.content


async def get_prediction_data(db: Session, product_id: int, window_size: int):
    try:
        # 1. 실제 가격 데이터 조회 (최신순 80일치 가져와서 날짜순 정렬)
        # 차트의 앞부분을 구성합니다.
        daily_prices = db.query(DailyPrice).filter(
            DailyPrice.product_id == product_id
        ).order_by(DailyPrice.base_date.desc()).limit(80).all()

        if not daily_prices:
            print(f"상품 ID {product_id}에 대한 실제 가격 데이터가 없습니다.")
            return None

        # 다시 날짜 오름차순으로 정렬
        daily_prices.reverse()
        latest_base_date = daily_prices[-1].base_date

        predictions = db.query(PredictPrice).filter(
            PredictPrice.product_id == product_id,
            PredictPrice.window_size == window_size,
            PredictPrice.base_date == latest_base_date
        ).order_by(PredictPrice.predict_date.asc()).limit(20).all()

        total_data = []
        
        # 실제 데이터 추가
        for p in daily_prices:
            total_data.append({
                "date": p.base_date.strftime('%Y-%m-%d'),
                "close": p.closing_price
            })
            
        # 예측 데이터 추가
        for p in predictions:
            total_data.append({
                "date": p.predict_date.strftime('%Y-%m-%d'),
                "close": p.predicted_close
            })

        return {
            "base_date": latest_base_date.strftime('%Y-%m-%d'),
            "window_size": window_size,
            "data": total_data  # 전체 100개의 데이터
        }

    except Exception as e:
        print(f"데이터 조회 중 예외 발생: {e}")
        return None