"""
src/telemetry/notifiers.py

Failure notification backends: Slack, Email, and log aggregation.
"""
from __future__ import annotations
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SlackNotifier:
    """
    Sends experiment completion/failure notifications to Slack.
    Requires SLACK_WEBHOOK_URL environment variable or explicit webhook_url.
    """

    def __init__(self, webhook_url: Optional[str] = None) -> None:
        import os
        self.webhook_url = webhook_url or os.environ.get("SLACK_WEBHOOK_URL")

    def notify(self, message: str, level: str = "info") -> bool:
        if not self.webhook_url:
            logger.warning("SlackNotifier: no webhook URL configured.")
            return False
        try:
            import requests
            emoji = {"info": ":information_source:", "error": ":x:",
                     "success": ":white_check_mark:"}.get(level, ":bell:")
            payload = {"text": f"{emoji} {message}"}
            resp = requests.post(self.webhook_url, json=payload, timeout=5)
            return resp.ok
        except Exception as exc:
            logger.error("Slack notification failed: %s", exc)
            return False


class EmailNotifier:
    """
    Email alerts for critical experiment failures via SMTP.
    Configure via environment: SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS.
    """

    def __init__(self, to_addr: str, from_addr: Optional[str] = None) -> None:
        import os
        self.to_addr   = to_addr
        self.from_addr = from_addr or os.environ.get("SMTP_USER", "noreply@localhost")
        self.smtp_host = os.environ.get("SMTP_HOST", "localhost")
        self.smtp_port = int(os.environ.get("SMTP_PORT", "587"))

    def notify(self, subject: str, body: str) -> bool:
        try:
            import smtplib
            from email.mime.text import MIMEText
            import os
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"]    = self.from_addr
            msg["To"]      = self.to_addr
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10) as server:
                user = os.environ.get("SMTP_USER")
                pw   = os.environ.get("SMTP_PASS")
                if user and pw:
                    server.starttls()
                    server.login(user, pw)
                server.send_message(msg)
            return True
        except Exception as exc:
            logger.error("Email notification failed: %s", exc)
            return False


class LogAggregator:
    """
    Collects logs from distributed workers to a central JSONL file.
    """

    def __init__(self, output_path: str) -> None:
        from src.telemetry.jsonl_storage import JSONLStorage
        self._store = JSONLStorage(output_path)

    def log(self, rank: int, step: int, message: str, level: str = "info") -> None:
        self._store.write({"rank": rank, "step": step, "message": message, "level": level})

    def close(self) -> None:
        self._store.close()
