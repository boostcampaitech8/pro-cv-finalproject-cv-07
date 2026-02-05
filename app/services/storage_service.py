import uuid
import requests
from io import BytesIO
from app.utils import upload_to_gcs


def backup_social_image(provider: str, provider_id: str, image_url: str):
    if not image_url:
        return None

    try:
        resp = requests.get(image_url)
        if resp.status_code != 200:
            return image_url

        image_file = BytesIO(resp.content)
        filename = (
            f"profiles/social_{provider}_{provider_id}_"
            f"{uuid.uuid4().hex[:8]}.jpg"
        )
        return upload_to_gcs(image_file, filename)

    except Exception as e:
        print(f"GCS 업로드 실패: {e}")
        return image_url