from sqlalchemy import Column, String, BigInteger, Integer, ForeignKey, DateTime, Date, UniqueConstraint, Float, Enum, JSON, func
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base, ReportBase


class User(Base):
    __tablename__ = "member"

    member_id = Column(BigInteger, primary_key=True, index=True, autoincrement=True)
    nickname = Column(String(100), unique=True, nullable=False)
    email = Column(String(255), nullable=True)
    birth_date = Column(Date, nullable=True) 
    gender = Column(String(10), nullable=True) 
    profile_image = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    social_accounts = relationship("UserSocialAccount", back_populates="user")
    favorites = relationship("Favorite", back_populates="user", cascade="all, delete-orphan")


class UserSocialAccount(Base):
    __tablename__ = "social_account"
    
    social_id = Column(BigInteger, primary_key=True, index=True, autoincrement=True)
    member_id = Column(BigInteger, ForeignKey("member.member_id"), nullable=False)
    provider_name = Column(String(20), nullable=False) # google, naver 등
    provider_user_identifier = Column(String(255), nullable=False) # 소셜 고유 ID
    email = Column(String(255), nullable=True)

    user = relationship("User", back_populates="social_accounts")

    __table_args__ = (
        UniqueConstraint("provider_name", "provider_user_identifier", name="uq_social_provider"),
    )


class Product(Base):
    __tablename__ = "product"

    product_id = Column(BigInteger, primary_key=True, index=True, autoincrement=True)
    product_name = Column(String(50), nullable=False)
    ticker = Column(String(20), nullable=False, unique=True)

    prices = relationship("DailyPrice", back_populates="product", cascade="all, delete-orphan")
    favorites = relationship("Favorite", back_populates="product", cascade="all, delete-orphan")
    news_list = relationship("News", back_populates="product", cascade="all, delete-orphan")
    predict_price = relationship("PredictPrice", back_populates="product", cascade="all, delete-orphan")


class DailyPrice(Base):
    __tablename__ = "daily_price"

    price_id = Column(BigInteger, primary_key=True, autoincrement=True, unique=True)
    product_id = Column(BigInteger, ForeignKey("product.product_id", ondelete="CASCADE"), nullable=False)
    base_date = Column(DateTime, nullable=False)
    closing_price = Column(Float, nullable=False)
    prev_closing_price = Column(Float, nullable=False)
    price_change = Column(Float, nullable=False)
    change_rate = Column(Float, nullable=False)

    product = relationship("Product", back_populates="prices")
    
    
class Favorite(Base):
    __tablename__ = "favorite"

    member_id = Column(BigInteger, ForeignKey("member.member_id", ondelete="CASCADE"), primary_key=True, nullable=False)
    product_id = Column(BigInteger, ForeignKey("product.product_id", ondelete="CASCADE"), primary_key=True, nullable=False)

    product = relationship("Product", back_populates="favorites")
    user = relationship("User", back_populates="favorites")
    
    
class News(Base):
    __tablename__ = "news"

    news_id = Column(BigInteger, primary_key=True, autoincrement=True)
    product_id = Column(BigInteger, ForeignKey("product.product_id", ondelete="CASCADE"), nullable=False)
    
    title = Column(String(255), nullable=False)
    news_url = Column(String(255), nullable=False)
    site_name = Column(String(255), nullable=False)
    published_at = Column(DateTime, nullable=False) 

    product = relationship("Product", back_populates="news_list")
    
    
class PredictPrice(Base):
    __tablename__ = "predict_price"

    # 컬럼 정의
    predict_id = Column(BigInteger, primary_key=True, autoincrement=True, nullable=False)
    product_id = Column(BigInteger, ForeignKey("product.product_id", ondelete="CASCADE"), nullable=False)
    
    base_date = Column(DateTime, nullable=False)
    window_size = Column(Integer, nullable=False)
    predict_date = Column(DateTime, nullable=False)
    predicted_close = Column(Float, nullable=False)

    product = relationship("Product", back_populates="predict_price")


# ── Report DB 모델 ──

class ReportUser(ReportBase):
    __tablename__ = "users"

    member_id = Column(BigInteger, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now())


class Conversation(ReportBase):
    __tablename__ = "conversations"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    member_id = Column(BigInteger, nullable=False, index=True)
    report_id = Column(BigInteger, ForeignKey("reports.id"), nullable=True)
    active_until_utc = Column(DateTime, nullable=True, index=True)
    status = Column(Enum('active', 'readonly', name='conversation_status'), nullable=False, server_default='active')
    locked_at_utc = Column(DateTime, nullable=True)
    last_message_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())
    item_type = Column(String(50), nullable=True)

    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")


class Report(ReportBase):
    __tablename__ = "reports"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    publish_date = Column(Date, nullable=False, index=True)
    keyword = Column(String(100), nullable=False, index=True)
    content = Column(LONGTEXT, nullable=False)
    meta = Column(JSON, nullable=True)
    images = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now())


class Message(ReportBase):
    __tablename__ = "messages"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id = Column(BigInteger, ForeignKey("conversations.id"), nullable=False, index=True)
    seq = Column(Integer, nullable=False)
    role = Column(Enum('system', 'user', 'assistant', 'tool', name='message_role'), nullable=False)
    content = Column(JSON, nullable=False)
    provider = Column(String(32), nullable=True)
    model = Column(String(64), nullable=True)
    token_in = Column(Integer, nullable=True)
    token_out = Column(Integer, nullable=True)

    conversation = relationship("Conversation", back_populates="messages")

