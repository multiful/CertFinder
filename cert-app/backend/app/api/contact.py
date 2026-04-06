import asyncio
import logging
import smtplib
import traceback
from email.header import Header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr

from app.api.deps import check_rate_limit
from app.config import get_settings

router = APIRouter(prefix="/contact", tags=["contact"])
logger = logging.getLogger(__name__)


class ContactRequest(BaseModel):
    name: str
    email: EmailStr
    subject: str
    message: str


class ContactEmailError(Exception):
    """문의 메일 발송 실패(클라이언트용 메시지 + HTTP 상태)."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def _send_one(
    server: smtplib.SMTP,
    from_addr: str,
    to_addr: str,
    subject: str,
    body: str,
):
    msg = MIMEMultipart()
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg["Subject"] = Header(subject, "utf-8")
    msg.attach(MIMEText(body, "plain", "utf-8"))
    server.send_message(msg)


def _send_contact_via_resend(
    settings,
    name: str,
    sender_email: str,
    subject: str,
    message: str,
) -> None:
    """Resend REST API (443). 클라우드에서 SMTP 587 차단될 때 사용."""
    key = (settings.RESEND_API_KEY or "").strip()
    if not key:
        raise ContactEmailError(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "메일 발송이 아직 설정되지 않았습니다. 잠시 후 다시 시도하거나 다른 경로로 문의해 주세요.",
        )

    admin_email = (settings.CONTACT_EMAIL or settings.EMAIL_USER or "").strip()
    if not admin_email:
        logger.error("[Resend] CONTACT_EMAIL 또는 EMAIL_USER 가 필요합니다 (관리자 수신 주소).")
        raise ContactEmailError(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "문의 수신 주소가 설정되지 않았습니다. 관리자에게 문의해 주세요.",
        )

    from_addr = (settings.RESEND_FROM_EMAIL or "").strip() or "CertFinder <onboarding@resend.dev>"

    to_admin = f"발신자: {name} ({sender_email})\n\n{message}"
    to_user = (
        f"{name}님, 문의해 주셔서 감사합니다.\n\n"
        "문의가 접수되었습니다. 검토 후 입력해 주신 이메일로 빠른 시일 내에 답변 드리겠습니다.\n\n"
        "— CertFinder"
    )

    logger.info(
        "[Resend] 발송 시도: from=%s → admin=%s, user=%s",
        from_addr,
        admin_email,
        sender_email,
    )

    url = "https://api.resend.com/emails"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    try:
        with httpx.Client(timeout=25.0) as client:
            r1 = client.post(
                url,
                headers=headers,
                json={
                    "from": from_addr,
                    "to": [admin_email],
                    "subject": f"[CertFinder 문의] {subject}",
                    "text": to_admin,
                },
            )
            if r1.status_code >= 400:
                logger.error("[Resend] 관리자 메일 실패: %s %s", r1.status_code, r1.text)
                raise ContactEmailError(
                    status.HTTP_502_BAD_GATEWAY,
                    "메일 발송에 실패했습니다. 잠시 후 다시 시도해 주세요.",
                )

            r2 = client.post(
                url,
                headers=headers,
                json={
                    "from": from_addr,
                    "to": [sender_email],
                    "subject": "[CertFinder] 문의 접수 완료",
                    "text": to_user,
                },
            )
            if r2.status_code >= 400:
                logger.error("[Resend] 사용자 확인 메일 실패: %s %s", r2.status_code, r2.text)
                raise ContactEmailError(
                    status.HTTP_502_BAD_GATEWAY,
                    "메일 발송에 실패했습니다. 잠시 후 다시 시도해 주세요.",
                )

        logger.info("[Resend] 발송 완료: admin(%s) + user(%s)", admin_email, sender_email)
    except ContactEmailError:
        raise
    except httpx.TimeoutException as e:
        logger.error("[Resend] 타임아웃: %s", e)
        raise ContactEmailError(
            status.HTTP_504_GATEWAY_TIMEOUT,
            "메일 발송 처리 시간이 초과되었습니다. 잠시 후 다시 시도해 주세요.",
        ) from e
    except httpx.HTTPError as e:
        logger.error("[Resend] HTTP 오류: %s", e)
        raise ContactEmailError(
            status.HTTP_502_BAD_GATEWAY,
            "메일 발송에 실패했습니다. 잠시 후 다시 시도해 주세요.",
        ) from e


def send_contact_emails_sync(name: str, sender_email: str, subject: str, message: str) -> None:
    """
    관리자·발신자에게 메일 전송. 실패 시 ContactEmailError.
    RESEND_API_KEY 가 있으면 Resend(HTTPS), 없으면 SMTP.
    (동기 함수 — 엔드포인트에서 asyncio.to_thread로 실행)
    """
    settings = get_settings()

    if (settings.RESEND_API_KEY or "").strip():
        _send_contact_via_resend(settings, name, sender_email, subject, message)
        return

    if not settings.EMAIL_USER or not settings.EMAIL_PASSWORD:
        logger.error(
            "[SMTP] 환경변수 미설정 — EMAIL_USER=%r, SMTP_HOST=%r. "
            "Railway Variables에 EMAIL_USER, EMAIL_PASSWORD, SMTP_HOST, SMTP_PORT, "
            "(선택) CONTACT_EMAIL 을 설정하세요.",
            settings.EMAIL_USER,
            settings.SMTP_HOST,
        )
        raise ContactEmailError(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "메일 발송이 아직 설정되지 않았습니다. 잠시 후 다시 시도하거나 다른 경로로 문의해 주세요.",
        )

    admin_email = (settings.CONTACT_EMAIL or settings.EMAIL_USER).strip()

    logger.info(
        "[SMTP] 메일 발송 시도: host=%s port=%d user=%s → admin=%s, user=%s",
        settings.SMTP_HOST,
        settings.SMTP_PORT,
        settings.EMAIL_USER,
        admin_email,
        sender_email,
    )

    to_admin = f"발신자: {name} ({sender_email})\n\n{message}"
    to_user = (
        f"{name}님, 문의해 주셔서 감사합니다.\n\n"
        "문의가 접수되었습니다. 검토 후 입력해 주신 이메일로 빠른 시일 내에 답변 드리겠습니다.\n\n"
        "— CertFinder"
    )

    try:
        if settings.SMTP_PORT == 465:
            server = smtplib.SMTP_SSL(settings.SMTP_HOST, settings.SMTP_PORT, timeout=15)
        else:
            server = smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=15)
            server.ehlo()
            server.starttls()
            server.ehlo()

        server.login(settings.EMAIL_USER, settings.EMAIL_PASSWORD)

        _send_one(
            server,
            settings.EMAIL_USER,
            admin_email,
            f"[CertFinder 문의] {subject}",
            to_admin,
        )
        _send_one(
            server,
            settings.EMAIL_USER,
            sender_email,
            "[CertFinder] 문의 접수 완료",
            to_user,
        )

        server.quit()
        logger.info("[SMTP] 발송 완료: admin(%s) + user(%s)", admin_email, sender_email)
    except ContactEmailError:
        raise
    except smtplib.SMTPAuthenticationError as e:
        logger.error(
            "[SMTP] 인증 실패 — 앱 비밀번호·SMTP 사용 설정을 확인하세요: %s",
            e,
        )
        raise ContactEmailError(
            status.HTTP_502_BAD_GATEWAY,
            "메일 서버 인증에 실패했습니다. 잠시 후 다시 시도해 주세요.",
        ) from e
    except smtplib.SMTPConnectError as e:
        logger.error(
            "[SMTP] 연결 실패 — host=%s port=%d: %s",
            settings.SMTP_HOST,
            settings.SMTP_PORT,
            e,
        )
        raise ContactEmailError(
            status.HTTP_502_BAD_GATEWAY,
            "메일 서버에 연결할 수 없습니다. 잠시 후 다시 시도해 주세요.",
        ) from e
    except smtplib.SMTPException as e:
        logger.error("[SMTP] SMTP 오류: %s", e)
        raise ContactEmailError(
            status.HTTP_502_BAD_GATEWAY,
            "메일 발송에 실패했습니다. 잠시 후 다시 시도해 주세요.",
        ) from e
    except OSError as e:
        logger.error("[SMTP] 네트워크 오류: %s", e)
        raise ContactEmailError(
            status.HTTP_502_BAD_GATEWAY,
            "메일 발송 중 네트워크 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
        ) from e
    except Exception as e:
        logger.error("[SMTP] 발송 실패: %s\n%s", e, traceback.format_exc())
        raise ContactEmailError(
            status.HTTP_502_BAD_GATEWAY,
            "메일 발송에 실패했습니다. 잠시 후 다시 시도해 주세요.",
        ) from e


# SMTP·네트워크가 멈추면 워커가 오래 점유되지 않도록 전체 상한(클라이언트 fetch 타임아웃과 맞출 것)
_CONTACT_SEND_TIMEOUT_SEC = 32.0


@router.post("")
async def submit_contact(
    request: ContactRequest,
    _: None = Depends(check_rate_limit),
):
    """문의 접수 — Resend(HTTPS) 또는 SMTP 전송 완료 후 성공(실패 시 502/503/504)."""
    try:
        await asyncio.wait_for(
            asyncio.to_thread(
                send_contact_emails_sync,
                request.name,
                str(request.email),
                request.subject,
                request.message,
            ),
            timeout=_CONTACT_SEND_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        logger.error(
            "[contact] 발송 시간 초과 (%.1fs) — SMTP/Resend 지연 가능",
            _CONTACT_SEND_TIMEOUT_SEC,
        )
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="메일 발송 처리 시간이 초과되었습니다. 네트워크를 확인한 뒤 잠시 후 다시 시도해 주세요.",
        ) from None
    except ContactEmailError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail) from e

    return {"message": "문의가 성공적으로 접수되었습니다."}
