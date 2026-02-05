from fastapi import Request, Depends, Form, File, UploadFile
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.main import templates
from app.services.profile_service import (
    get_user,
    is_nickname_available,
    update_profile,
    upload_profile_image
)


async def edit_profile_page(
    request: Request,
    db: Session = Depends(get_db),
):
    user_id = request.session.get("user_id")
    if not user_id:
        return RedirectResponse(url="/login")

    user = get_user(db, user_id)

    if user and user.profile_image:
        user.profile_image = (
            str(user.profile_image)
            .replace("b'", "")
            .replace("'", "")
        )

    return templates.TemplateResponse(
        "edit_profile.html",
        {
            "request": request,
            "user": user,
        },
    )


async def process_edit_profile(
    request: Request,
    nickname: str = Form(...),
    birth_date: str = Form(None),
    gender: str = Form(None),
    profile_pic: UploadFile = File(None),
    db: Session = Depends(get_db),
):
    user_id = request.session.get("user_id")
    user = get_user(db, user_id)

    if not is_nickname_available(db, nickname, user_id):
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "title": "닉네임 중복",
                "message": "이미 사용 중인 닉네임입니다.",
            },
            status_code=400,
        )

    profile_image_url = None
    if profile_pic and profile_pic.filename:
        try:
            profile_image_url = upload_profile_image(user_id, profile_pic)
        except Exception as e:
            print(f"GCS 업로드 실패: {e}")

    try:
        update_profile(
            db,
            user=user,
            nickname=nickname,
            birth_date=birth_date,
            gender=gender,
            new_profile_image=profile_image_url,
        )

        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "icon": "✅",
                "title": "수정 완료",
                "message": "회원 정보가 성공적으로 변경되었습니다.",
                "is_success": True,
                "button_text": "홈으로 돌아가기",
            },
        )

    except Exception as e:
        db.rollback()
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "icon": "❌",
                "title": "오류 발생",
                "message": "회원 정보 수정 중 문제가 발생했습니다.",
                "is_success": False,
                "button_text": "이전으로",
            },
            status_code=500,
        )
