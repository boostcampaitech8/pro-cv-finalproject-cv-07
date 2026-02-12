import uuid
from sqlalchemy.orm import Session
from app import models
from app.utils import delete_from_gcs, upload_to_gcs


def get_user(db: Session, user_id: int):
    return (
        db.query(models.User)
        .filter(models.User.member_id == user_id)
        .first()
    )


def is_nickname_available(
    db: Session,
    nickname: str,
    current_user_id: int,
) -> bool:
    return (
        db.query(models.User)
        .filter(
            models.User.nickname == nickname,
            models.User.member_id != current_user_id,
        )
        .first()
        is None
    )


def update_profile(
    db: Session,
    *,
    user: models.User,
    nickname: str,
    birth_date: str,
    gender: str,
    new_profile_image: str | None = None,
):
    user.nickname = nickname
    user.birth_date = birth_date
    user.gender = gender

    if new_profile_image:
        # 기존 이미지 삭제
        if user.profile_image:
            try:
                delete_from_gcs(user.profile_image)
            except Exception as e:
                print(f"기존 프로필 이미지 삭제 실패: {e}")

        # 새 이미지 반영
        user.profile_image = new_profile_image

    db.commit()
    
    
def upload_profile_image(user_id: int, profile_pic) -> str:
    ext = profile_pic.filename.split('.')[-1].lower()
    filename = f"profiles/{user_id}_{uuid.uuid4().hex}.{ext}"

    file_obj = profile_pic.file
    return upload_to_gcs(file_obj, filename)