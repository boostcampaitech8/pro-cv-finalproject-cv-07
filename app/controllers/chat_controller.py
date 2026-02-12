from fastapi import Request, Depends, HTTPException
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from app.main import templates
from app.database import get_db, get_report_db
from pydantic import BaseModel
from app.services.chatbot_service.chat_service import create_conversation, get_conversation, get_conversation_messages, send_chat_message, get_report_for_chat
from app.models import Product


async def chat_loading(request: Request, product_id: int, db: Session = Depends(get_db), report_db: Session = Depends(get_report_db)):
    user_id = request.session.get("user_id")
    if not user_id:
        return RedirectResponse(url="/login", status_code=303)

    product = db.query(Product).filter(Product.product_id == product_id).first()
    item_type = product.ticker if product else None

    conversation = await create_conversation(report_db, member_id=user_id, item_type=item_type)

    return templates.TemplateResponse("chat_loading.html", {
        "request": request,
        "conversation_id": conversation.id,
        "product_name": product.product_name if product else "",
    })


async def chat_page(conversation_id: int, request: Request, report_db: Session = Depends(get_report_db)):
    user_id = request.session.get("user_id")
    if not user_id:
        return RedirectResponse(url="/login", status_code=303)

    conversation = await get_conversation(report_db, conversation_id)
    if not conversation or conversation.member_id != user_id:
        raise HTTPException(status_code=404, detail="대화를 찾을 수 없습니다.")

    messages = await get_conversation_messages(report_db, conversation.id)

    # 리포트 내용 조회
    report_content = ""
    if conversation.report_id:
        from app.models import Report
        report = report_db.query(Report).filter(Report.id == conversation.report_id).first()
        if report and report.content:
            report_content = report.content

    return templates.TemplateResponse("chat.html", {
        "request": request,
        "conversation_id": conversation.id,
        "item_type": conversation.item_type or "",
        "messages": messages,
        "report_content": report_content,
    })


class ChatMessageRequest(BaseModel):
    conversation_id: int
    message: str


async def send_message(
    data: ChatMessageRequest,
    request: Request,
    report_db: Session = Depends(get_report_db)
):
    user_id = request.session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")

    conversation = await get_conversation(report_db, data.conversation_id)
    if not conversation or conversation.member_id != user_id:
        raise HTTPException(status_code=404, detail="대화를 찾을 수 없습니다.")

    result = await send_chat_message(report_db, data.conversation_id, data.message)
    if result is None:
        raise HTTPException(status_code=500, detail="응답 생성에 실패했습니다.")

    return {
        "reply": result["reply"],
        "used_graph": result["used_graph"],
        "sources": result.get("sources", []),
    }
