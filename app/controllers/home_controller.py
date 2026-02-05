from fastapi import Request, Depends
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from app.services.user_service import get_user_by_session
from app.database import get_db
from app.main import templates


async def home(request: Request, db: Session = Depends(get_db)):
    user_id = request.session.get("user_id")
    user = get_user_by_session(db, user_id)

    return templates.TemplateResponse(
        "home.html",
        {"request": request, "user": user}
    )