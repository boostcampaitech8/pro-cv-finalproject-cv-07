import logging, os
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from authlib.integrations.starlette_client import OAuth
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv

from app.database import engine
import app.models as models


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SESSION_SECRET", "very-secret-key"))

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

# router 등록
import app.routers as routers
app.include_router(routers.router)