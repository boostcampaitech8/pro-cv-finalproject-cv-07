from fastapi import Request, Depends, Form, HTTPException, status
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from app.main import templates, oauth
from app.database import get_db
from app.services.oauth_service import get_oauth_user_info
from app.services.storage_service import backup_social_image
from app.services.user_service import (
    get_social_account,
    is_nickname_taken,
    create_user_with_social,
)
from app import models


DEFAULT_PROFILE_IMAGE = "https://storage.googleapis.com/web-service-predict/profiles/default-profile.jpg"


async def login_page(request: Request):
    return templates.TemplateResponse(
        "login.html",
        {"request": request}
    )


async def login(provider: str, request: Request):
    redirect_uri = request.url_for("oauth_callback", provider=provider)
    client = oauth.create_client(provider)
    
    if provider == "google":
        return await client.authorize_redirect(
            request, redirect_uri, prompt="login"
        )

    if provider == "naver":
        return await client.authorize_redirect(
            request, redirect_uri, auth_type="reauthenticate"
        )
    
    return await client.authorize_redirect(request, redirect_uri)


async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/")


async def oauth_callback(
    provider: str,
    request: Request,
    db: Session = Depends(get_db),
):
    client = oauth.create_client(provider)
    token = await client.authorize_access_token(request)
    user_info = await get_oauth_user_info(client, provider, token)

    social_image = user_info.get("profile_pic")

    if social_image:
        gcs_profile = backup_social_image(
            provider,
            user_info["provider_id"],
            social_image
        )
    else:
        gcs_profile = DEFAULT_PROFILE_IMAGE

    social_acc = get_social_account(
        db, provider, user_info["provider_id"]
    )

    if social_acc:
        request.session["user_id"] = social_acc.member_id
        return RedirectResponse(url="/")

    request.session["temp_social_info"] = {
        **user_info,
        "provider": provider,
        "profile_image": gcs_profile,
    }

    return RedirectResponse(url="/extra-info")


async def extra_info_page(request: Request):
    temp_info = request.session.get("temp_social_info")
    if not temp_info:
        return RedirectResponse(url="/")

    return templates.TemplateResponse(
        "extra_info.html",
        {"request": request, "temp_info": temp_info},
    )
    

async def save_extra_info(
    request: Request,
    nickname: str = Form(...),
    birth_date: str = Form(...),
    gender: str = Form(...),
    db: Session = Depends(get_db),
):
    temp = request.session.get("temp_social_info")
    if not temp:
        raise HTTPException(400, "세션 만료")

    if is_nickname_taken(db, nickname):
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "title": "닉네임 중복",
                "message": "이미 사용 중인 닉네임입니다.",
            },
            status_code=400,
        )

    user = create_user_with_social(
        db,
        nickname=nickname,
        email=temp["email"],
        profile_image=temp["profile_image"],
        birth_date=birth_date,
        gender=gender,
        provider=temp["provider"],
        provider_id=temp["provider_id"],
    )

    del request.session["temp_social_info"]
    request.session["user_id"] = user.member_id

    return RedirectResponse(
        url="/welcome",
        status_code=status.HTTP_303_SEE_OTHER,
    )
    
    
async def welcome(
    request: Request,
    db: Session = Depends(get_db),
):
    user_id = request.session.get("user_id")
    if not user_id:
        return RedirectResponse(url="/")

    user = (
        db.query(models.User)
        .filter(models.User.member_id == user_id)
        .first()
    )

    return templates.TemplateResponse(
        "welcome.html",
        {"request": request, "user": user},
    )