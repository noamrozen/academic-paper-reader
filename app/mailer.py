"""Sends the account emails (verify an address, reset a password): through Resend, SMTP, or, with
neither configured, only to the log (handy locally: the link is printed in the terminal)."""

import asyncio
import logging
import smtplib
from email.message import EmailMessage

import httpx

from .config import EMAIL_FROM, RESEND_API_KEY, SMTP_HOST, SMTP_PASSWORD, SMTP_PORT, SMTP_USERNAME

log = logging.getLogger(__name__)


def configured() -> bool:
    return bool(RESEND_API_KEY or SMTP_HOST)


async def send(to: str, subject: str, text: str):
    if RESEND_API_KEY:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post("https://api.resend.com/emails", json={
                "from": EMAIL_FROM, "to": [to], "subject": subject, "text": text,
            }, headers={"Authorization": f"Bearer {RESEND_API_KEY}"})
        if r.is_error:
            raise RuntimeError(f"Resend: {r.text[:200]}")
    elif SMTP_HOST:
        await asyncio.to_thread(_smtp, to, subject, text)
    else:
        log.warning("No email service configured (RESEND_API_KEY or SMTP_HOST). Email to %s: %s\n%s",
                    to, subject, text)


def _smtp(to: str, subject: str, text: str):
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = EMAIL_FROM, to, subject
    msg.set_content(text)
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as smtp:
        smtp.starttls()
        if SMTP_USERNAME:
            smtp.login(SMTP_USERNAME, SMTP_PASSWORD or "")
        smtp.send_message(msg)
