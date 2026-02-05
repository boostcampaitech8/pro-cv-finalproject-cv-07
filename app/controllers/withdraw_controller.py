from fastapi import Request, Depends
from fastapi.responses import RedirectResponse, HTMLResponse
from sqlalchemy.orm import Session

from app.main import oauth, templates
from app.database import get_db
from app.services.oauth_service import get_oauth_user_info
from app.services.withdraw_service import (
    verify_social_account_for_withdraw,
    withdraw_user,
)
from app import models


async def withdraw_verify(provider: str, request: Request):
    redirect_uri = request.url_for("auth_withdraw", provider=provider)
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


async def auth_withdraw(
    provider: str,
    request: Request,
    db: Session = Depends(get_db),
):
    try:
        client = oauth.create_client(provider)
        token = await client.authorize_access_token(request)

        user_info = await get_oauth_user_info(client, provider, token)
        current_user_id = request.session.get("user_id")

        if not current_user_id:
            return RedirectResponse(url="/")

        social_acc = verify_social_account_for_withdraw(
            db,
            member_id=current_user_id,
            provider=provider,
            provider_user_identifier=user_info["provider_id"],
        )

        if not social_acc:
            return templates.TemplateResponse(
                "result.html",
                {
                    "request": request,
                    "icon": "🔒",
                    "title": "인증 정보 불일치",
                    "message": (
                        "현재 로그인된 계정과 인증하신 소셜 계정이 일치하지 않습니다."
                        "<br>본인 계정으로 다시 시도해주세요."
                    ),
                    "is_success": False,
                    "button_text": "홈으로 돌아가기",
                },
            )

        withdraw_user(db, current_user_id)
        request.session.clear()

        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "icon": "👋",
                "title": "탈퇴 완료",
                "message": (
                    "그동안 서비스를 이용해 주셔서 감사합니다."
                    "<br>모든 정보가 안전하게 삭제되었습니다."
                ),
                "is_success": True,
            },
        )

    except Exception:
        db.rollback()
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "icon": "⚠️",
                "title": "오류 발생",
                "message": (
                    "탈퇴 처리 중 예상치 못한 오류가 발생했습니다."
                    "<br>잠시 후 다시 시도해주세요."
                ),
                "is_success": False,
                "button_text": "홈으로 돌아가기",
            },
        )


async def withdraw_page(
    request: Request,
    db: Session = Depends(get_db),
):
    user_id = request.session.get("user_id")
    if not user_id:
        return RedirectResponse(url="/")

    social_accs = (
        db.query(models.UserSocialAccount)
        .filter(models.UserSocialAccount.member_id == user_id)
        .all()
    )

    return templates.TemplateResponse(
        "withdraw.html",
        {
            "request": request,
            "social_accs": social_accs,
        },
    )
