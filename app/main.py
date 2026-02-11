import logging, os
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from authlib.integrations.starlette_client import OAuth
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv

from app.database import engine, report_engine
import app.models as models


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    SessionMiddleware, 
    secret_key=os.getenv("SESSION_SECRET", "very-secret-key"),
    max_age=10800  # 3시간 (3 * 3600초)
)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

oauth = OAuth()
oauth.register(
    name='google',
    client_id=os.getenv('GOOGLE_CLIENT_ID'),
    client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)
oauth.register(
    name='naver',
    client_id=os.getenv('NAVER_CLIENT_ID'),
    client_secret=os.getenv('NAVER_CLIENT_SECRET'),
    authorize_url='https://nid.naver.com/oauth2.0/authorize',
    access_token_url='https://nid.naver.com/oauth2.0/token',
    userinfo_endpoint='https://openapi.naver.com/v1/nid/me',
    client_kwargs={'scope': 'name email profile_image'}
)

templates = Jinja2Templates(directory="app/views")

models.Base.metadata.create_all(bind=engine)
try:
    models.ReportBase.metadata.create_all(bind=report_engine)
except Exception as e:
    logger.warning(f"Report DB 테이블 생성 실패 (무시하고 계속): {e}")

# router 등록
import app.routers as routers
app.include_router(routers.router)