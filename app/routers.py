from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from app.controllers import home_controller, auth_controller, profile_controller, withdraw_controller, product_controller, chat_controller

router = APIRouter()

# home
router.get("/")(home_controller.home_page)
router.get("/profile", response_class=HTMLResponse)(profile_controller.home)

router.get("/product/{product_id}")(product_controller.product_detail_page)
router.post("/api/favorite")(product_controller.toggle_favorite)
router.get("/api/prediction/{product_id}")(product_controller.read_prediction)

# login-logout
router.get("/login")(auth_controller.login_page)
router.get("/login/{provider}")(auth_controller.login)
router.get("/logout")(auth_controller.logout)

# 인증 및 회원가입
router.get("/oauth/{provider}")(auth_controller.oauth_callback)
router.get("/extra-info")(auth_controller.extra_info_page)
router.post("/extra-info")(auth_controller.save_extra_info)
router.get("/welcome")(auth_controller.welcome)

# 회원 정보 수정
router.get("/edit-profile")(profile_controller.edit_profile_page)
router.post("/edit-profile")(profile_controller.process_edit_profile)

# 회원 탈퇴
router.get("/withdraw")(withdraw_controller.withdraw_page)
router.get("/withdraw/verify/{provider}")(withdraw_controller.withdraw_verify)
router.get("/oauth/withdraw/{provider}")(withdraw_controller.auth_withdraw)

# 채팅
router.get("/chat-loading")(chat_controller.chat_loading)
router.get("/chat/{conversation_id}")(chat_controller.chat_page)
router.post("/api/chat/message")(chat_controller.send_message)