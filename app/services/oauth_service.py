import requests
from authlib.integrations.starlette_client import OAuth


async def get_oauth_user_info(client: OAuth, provider: str, token: dict) -> dict:
    user_data = {}
    social_pic_url = None

    if provider == "google":
        user_data = token.get("userinfo")
        social_pic_url = user_data.get("picture")
        provider_id = user_data.get("sub")
        email = user_data.get("email")
        nickname = user_data.get("name")
    elif provider == "naver":
        resp = await client.get("https://openapi.naver.com/v1/nid/me", token=token)
        user_data = resp.json().get("response", {})
        social_pic_url = user_data.get("profile_image")
        provider_id = user_data.get("id")
        email = user_data.get("email")
        nickname = user_data.get("nickname") or user_data.get("name")
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return {
        "provider_id": str(provider_id),
        "email": email,
        "nickname": nickname,
        "profile_pic": social_pic_url,
    }