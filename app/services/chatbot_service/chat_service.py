import os
import logging
from openai import OpenAI
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models import Conversation, Report, ReportUser, Message
from app.services.chatbot_service.graph_rag_service import retrieve_graph_context
from datetime import date

logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def ensure_report_user(report_db: Session, member_id: int):
    user = report_db.query(ReportUser).filter(ReportUser.member_id == member_id).first()
    if not user:
        user = ReportUser(member_id=member_id)
        report_db.add(user)
        report_db.commit()


async def create_conversation(report_db: Session, member_id: int, item_type: str = None):
    await ensure_report_user(report_db, member_id)

    report = await get_report_for_chat(report_db, item_type) if item_type else None
    report_id = report.id if report else None

    # 같은 user + report 조합의 active 대화가 있으면 재사용
    if report_id:
        existing = report_db.query(Conversation).filter(
            Conversation.member_id == member_id,
            Conversation.report_id == report_id,
            Conversation.status == 'active'
        ).first()
        if existing:
            return existing

    conversation = Conversation(
        member_id=member_id,
        report_id=report_id,
        item_type=item_type,
        status='active'
    )
    report_db.add(conversation)
    report_db.commit()
    report_db.refresh(conversation)
    return conversation


async def get_conversation_messages(report_db: Session, conversation_id: int):
    """대화의 모든 메시지를 순서대로 반환"""
    messages = report_db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.seq.asc()).all()
    result = []
    for msg in messages:
        content = msg.content if isinstance(msg.content, dict) else {"text": msg.content}
        item = {
            "role": msg.role,
            "text": content.get("text", ""),
        }
        if msg.role == "assistant":
            item["used_graph"] = content.get("used_graph", False)
            item["sources"] = content.get("sources", [])
        result.append(item)
    return result


async def get_conversation(report_db: Session, conversation_id: int):
    return report_db.query(Conversation).filter(Conversation.id == conversation_id).first()


async def get_report_for_chat(report_db: Session, ticker: str):
    today = date.today()
    report = report_db.query(Report).filter(
        Report.keyword == ticker,
        Report.publish_date == today
    ).first()
    return report


_GRAPH_KEYWORDS = [
    "기사", "뉴스", "소식", "동향", "이슈", "트렌드",
    "최근", "요즘", "근래", "올해", "이번 달", "이번 주",
    "왜 올랐", "왜 내렸", "왜 떨어", "왜 상승", "왜 하락",
    "원인", "배경", "영향", "전망", "분석",
    "사건", "이벤트", "정책", "규제", "관세",
    "수급", "공급", "수요", "재고", "생산",
]


def _should_use_graph(user_message: str, report_content: str = None) -> bool:
    """라우팅 모델: 사용자 질문이 GraphDB 검색이 필요한지 판단"""
    # 키워드 기반 빠른 판단: 기사/뉴스 관련 키워드가 있으면 무조건 GRAPH
    msg_lower = user_message.lower()
    if any(kw in msg_lower for kw in _GRAPH_KEYWORDS):
        logger.info(f"[라우팅] 키워드 매칭으로 GRAPH 결정")
        return True

    report_summary = ""
    if report_content:
        report_summary = report_content[:500]

    routing_prompt = f"""당신은 원자재 채팅봇의 라우팅 판단기입니다.
사용자의 질문을 보고, 외부 뉴스/기사 데이터베이스(GraphDB) 검색이 필요한지 판단하세요.

판단 기준: 의심스러우면 GRAPH를 선택하세요. DIRECT는 확실히 검색이 필요 없을 때만 선택합니다.

GraphDB 검색이 필요한 경우 (GRAPH):
- 뉴스, 기사, 시장 동향, 소식에 대한 질문
- 가격 변동의 원인이나 배경을 묻는 질문
- 특정 이벤트나 정책이 시장에 미치는 영향 질문
- 리포트에 없는 추가 정보가 필요한 질문
- 시장 상황, 전망, 분석에 대한 질문
- 예시나 사례를 요청하는 질문
- 수급, 공급, 수요, 재고 관련 질문

GraphDB 검색이 필요 없는 경우 (DIRECT):
- 간단한 인사, 감사 표현
- 리포트 내용을 요약하거나 설명해달라는 질문
- 일반적인 원자재 상식 질문 (정의, 개념 등)

현재 리포트 요약:
{report_summary}

사용자 질문: {user_message}

"GRAPH" 또는 "DIRECT" 중 하나만 답하세요."""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": routing_prompt}],
            max_completion_tokens=10,
        )
        answer = response.choices[0].message.content.strip().upper()
        return "GRAPH" in answer
    except Exception as e:
        logger.warning(f"라우팅 판단 실패, DIRECT로 진행: {e}")
        return False


async def send_chat_message(report_db: Session, conversation_id: int, user_message: str):
    conversation = report_db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conversation:
        return None

    # 리포트 content를 시스템 프롬프트로 사용 (정적 context)
    report = None
    report_content = None
    if conversation.report_id:
        report = report_db.query(Report).filter(Report.id == conversation.report_id).first()
        if report:
            report_content = report.content

    system_content = (
        "당신은 원자재 분석 전문 AI 어시스턴트 '주니'입니다. 사용자의 질문에 친절하고 정확하게 답변하세요.\n"
        "당신은 관련 뉴스와 기사 데이터에 접근할 수 있습니다. "
        "시스템 메시지로 뉴스/기사 정보가 제공되면 반드시 그 내용을 기반으로 답변하세요. "
        "\"실시간 조회가 불가능하다\"거나 \"웹 검색을 할 수 없다\"는 말은 하지 마세요."
    )
    if report_content:
        system_content += f"\n\n다음은 오늘의 분석 리포트입니다. 이 내용을 기반으로 답변하세요:\n\n{report_content}"

    # 기존 메시지 히스토리 조회 (최근 10개만: user 5 + assistant 5)
    existing_messages = report_db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.seq.desc()).limit(10).all()
    existing_messages.reverse()

    # 라우팅 판단: GraphDB 검색 필요 여부
    use_graph = _should_use_graph(user_message, report_content)
    logger.info(f"[라우팅] 질문='{user_message[:50]}' → {'GRAPH' if use_graph else 'DIRECT'}")

    graph_context = ""
    graph_sources = []
    if use_graph and conversation.item_type:
        graph_result = retrieve_graph_context(user_message, conversation.item_type)
        graph_context = graph_result["context"]
        graph_sources = graph_result["sources"]
        logger.info(f"[GraphRAG] item_type={conversation.item_type}, context길이={len(graph_context)}, sources={len(graph_sources)}")

    # OpenAI 메시지 배열 구성
    # [system: 역할+리포트] → [history] → [system: 그래프context (있으면)] → [user: 현재질문]
    openai_messages = [{"role": "system", "content": system_content}]

    # 히스토리 (순수 대화만)
    for msg in existing_messages:
        content = msg.content if isinstance(msg.content, str) else msg.content.get("text", "")
        openai_messages.append({"role": msg.role, "content": content})

    # 동적 context: GraphDB 트리플 (현재 턴 한정)
    if graph_context:
        openai_messages.append({
            "role": "system",
            "content": f"다음은 사용자의 질문과 관련된 최근 뉴스/기사에서 추출한 핵심 정보입니다. 답변 시 참고하세요:\n\n{graph_context}"
        })

    openai_messages.append({"role": "user", "content": user_message})

    # 다음 seq 번호
    max_seq = report_db.query(func.max(Message.seq)).filter(
        Message.conversation_id == conversation_id
    ).scalar() or 0

    # 유저 메시지 저장
    user_msg = Message(
        conversation_id=conversation_id,
        seq=max_seq + 1,
        role="user",
        content={"text": user_message},
    )
    report_db.add(user_msg)

    # OpenAI API 호출
    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=openai_messages,
        max_completion_tokens=1024,
    )

    assistant_text = response.choices[0].message.content
    token_in = response.usage.prompt_tokens if response.usage else None
    token_out = response.usage.completion_tokens if response.usage else None

    # 어시스턴트 메시지 저장
    assistant_msg = Message(
        conversation_id=conversation_id,
        seq=max_seq + 2,
        role="assistant",
        content={"text": assistant_text, "used_graph": bool(graph_context), "sources": graph_sources},
        provider="openai",
        model="gpt-5.2",
        token_in=token_in,
        token_out=token_out,
    )
    report_db.add(assistant_msg)

    # conversation 업데이트
    conversation.last_message_at = func.now()
    report_db.commit()

    return {"reply": assistant_text, "used_graph": bool(graph_context), "sources": graph_sources}
