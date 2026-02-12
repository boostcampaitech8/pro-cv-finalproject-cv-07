from fastapi import Request, Depends, HTTPException
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.services.product_service import get_product_detail, toggle_favorite_status, get_prediction_data, get_report_by_ticker
from app.main import templates
from app.database import get_db, get_report_db


async def product_detail_page(product_id: int, request: Request, db: Session = Depends(get_db), report_db: Session = Depends(get_report_db)):
    user_id = request.session.get("user_id")

    data = await get_product_detail(db, product_id, user_id)

    if not data:
        return RedirectResponse(url="/login", status_code=303)

    report_content = await get_report_by_ticker(report_db, data["ticker"])
    data["report_content"] = report_content

    return templates.TemplateResponse("product_detail.html", {
        "request": request,
        **data
    })
    

class FavoriteToggleRequest(BaseModel):
    product_id: int
    is_favorite: bool

async def toggle_favorite(
    data: FavoriteToggleRequest,
    request: Request,
    db: Session = Depends(get_db)
):
    user_id = request.session.get("user_id")
    
    if not user_id:
        raise HTTPException(status_code=401, detail="로그인이 필요한 서비스입니다.")

    success = await toggle_favorite_status(
        db, 
        member_id=user_id, 
        product_id=data.product_id, 
        is_favorite=data.is_favorite
    )
    
    if not success:
        raise HTTPException(status_code=500, detail="데이터베이스 처리 중 오류가 발생했습니다.")
    
    return {"status": "success", "current_state": data.is_favorite}


async def read_prediction(
    product_id: int, 
    window_size: int = 5, 
    db: Session = Depends(get_db)
):
    # Controller 함수 호출
    result = await get_prediction_data(db, product_id, window_size)
    
    if result is None:
        # 데이터가 없을 때 404를 내보내면 디버깅이 더 쉬워집니다.
        raise HTTPException(status_code=404, detail="Prediction data not found")
        
    return result