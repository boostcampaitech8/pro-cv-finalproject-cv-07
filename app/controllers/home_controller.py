from fastapi import Request, Depends
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from app.services.user_service import get_user_by_session
from app.services.home_service import get_home_data
from app.database import get_db
from app.main import templates
    

async def home_page(request: Request, db: Session = Depends(get_db)):
    user_id = request.session.get("user_id")
    
    user = get_user_by_session(db, user_id)
    data = await get_home_data(db, user_id)
    
    if user:
        data["is_logged_in"] = True
        data["user_name"] = user.nickname
        data["user_profile_img"] = user.profile_image
    else:
        data["is_logged_in"] = False
        data["user_name"] = "Guest"
    
    return templates.TemplateResponse("product-home.html", {"request": request, **data})