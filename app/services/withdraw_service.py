from sqlalchemy.orm import Session
from app import models
from app.utils import delete_from_gcs


DEFAULT_PROFILE_IMAGE = "https://storage.googleapis.com/web-service-predict/profiles/default-profile.jpg"


def verify_social_account_for_withdraw(
    db: Session,
    *,
    member_id: int,
    provider: str,
    provider_user_identifier: str,
):
    return (
        db.query(models.UserSocialAccount)
        .filter(
            models.UserSocialAccount.member_id == member_id,
            models.UserSocialAccount.provider_name == provider,
            models.UserSocialAccount.provider_user_identifier == provider_user_identifier,
        )
        .first()
    )


def withdraw_user(db: Session, member_id: int):
    user = (
        db.query(models.User)
        .filter(models.User.member_id == member_id)
        .first()
    )

    if not user:
        return False
    
    if user.profile_image and user.profile_image != DEFAULT_PROFILE_IMAGE:
        try:
            delete_from_gcs(user.profile_image)
        except Exception as e:
            print(f"GCS 이미지 삭제 실패: {e}")

    db.query(models.UserSocialAccount).filter(
        models.UserSocialAccount.member_id == member_id
    ).delete()

    db.delete(user)
    db.commit()
    return True
