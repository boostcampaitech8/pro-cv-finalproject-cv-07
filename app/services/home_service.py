from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models import Product, DailyPrice, Favorite


async def get_home_data(db: Session, user_id):
    latest_date_subquery = db.query(
        DailyPrice.product_id,
        func.max(DailyPrice.base_date).label('latest_date')
    ).group_by(DailyPrice.product_id).subquery()

    products_with_price = db.query(Product, DailyPrice).join(
        DailyPrice, Product.product_id == DailyPrice.product_id
    ).join(
        latest_date_subquery,
        (DailyPrice.product_id == latest_date_subquery.c.product_id) &
        (DailyPrice.base_date == latest_date_subquery.c.latest_date)
    ).all()

    emoji_map = {
        "ZC=F": "🌽", "ZW=F": "🌾", "ZS=F": "🫘",
        "GC=F": "💰", "SI=F": "🪙", "HG=F": "🧱"
    }
        
    fav_ids = []
    if user_id:
        fav_records = db.query(Favorite.product_id).filter(Favorite.member_id == user_id).all()
        fav_ids = [f.product_id for f in fav_records]

    all_products = []
    favorites = []
    not_favorites = []

    for p, price in products_with_price:
        item = {
            "id": p.product_id,
            "name_ko": p.product_name,
            "ticker": p.ticker,
            "emoji": emoji_map.get(p.ticker, "📦"),
            "price": f"${price.closing_price:,.2f}",
            "change_rate": f"{price.change_rate:+.2f}%",
            "is_positive": price.change_rate >= 0
        }
        all_products.append(item)
            
        # 즐겨찾기 목록에 추가
        if p.product_id in fav_ids:
            favorites.append(item)
        else:
            not_favorites.append(item)

    return {
        "all_products": not_favorites,
        "favorites": favorites
    }