from sqlalchemy import Column, String, BigInteger, ForeignKey, DateTime, Date, UniqueConstraint
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base


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