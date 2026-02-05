from sqlalchemy.orm import Session
from app import models


def get_user_by_session(db: Session, user_id: int):
    if not user_id:
        return None

    user = (
        db.query(models.User)
        .filter(models.User.member_id == user_id)
        .first()
    )

    if user and user.profile_image:
        user.profile_image = (
            str(user.profile_image)
            .replace("b'", "")
            .replace("'", "")
        )

    return user


def get_social_account(db: Session, provider: str, provider_id: str):
    return (
        db.query(models.UserSocialAccount)
        .filter(
            models.UserSocialAccount.provider_name == provider,
            models.UserSocialAccount.provider_user_identifier == provider_id
        )
        .first()
    )


def is_nickname_taken(db: Session, nickname: str) -> bool:
    return (
        db.query(models.User)
        .filter(models.User.nickname == nickname)
        .first()
        is not None
    )


def create_user_with_social(
    db: Session,
    *,
    nickname: str,
    email: str,
    profile_image: str,
    birth_date: str,
    gender: str,
    provider: str,
    provider_id: str,
):
    user = models.User(
        nickname=nickname,
        email=email,
        profile_image=profile_image,
        birth_date=birth_date,
        gender=gender,
    )
    db.add(user)
    db.flush()

    social = models.UserSocialAccount(
        member_id=user.member_id,
        provider_name=provider,
        provider_user_identifier=provider_id,
        email=email,
    )
    db.add(social)
    db.commit()

    return user