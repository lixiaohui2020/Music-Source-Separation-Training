from __future__ import annotations

import html
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from zoneinfo import ZoneInfo

from scripts.paper_digest.arxiv_client import Paper
from scripts.paper_digest.config import PaperDigestConfig


def _format_authors(authors: list[str], max_count: int = 5) -> str:
    if not authors:
        return "未知"
    if len(authors) <= max_count:
        return ", ".join(authors)
    return ", ".join(authors[:max_count]) + f" 等 {len(authors)} 人"


def build_html_email(
    papers: list[tuple[Paper, str, list[str]]],
    cfg: PaperDigestConfig,
) -> str:
    now = datetime.now(ZoneInfo(cfg.timezone))
    date_str = now.strftime("%Y年%m月%d日")

    if not papers:
        body = "<p>今日暂无新的人声/伴奏分离相关论文。</p>"
    else:
        sections = []
        for index, (paper, summary, github_links) in enumerate(papers, start=1):
            gh_html = ""
            if github_links:
                gh_items = "".join(
                    f'<li><a href="{html.escape(url)}">{html.escape(url)}</a></li>'
                    for url in github_links
                )
                gh_html = f"<p><strong>参考 GitHub：</strong></p><ul>{gh_items}</ul>"
            else:
                gh_html = "<p><strong>参考 GitHub：</strong>暂未找到公开仓库</p>"

            sections.append(
                f"""
                <div style="margin-bottom:24px;padding:16px;border:1px solid #e5e7eb;border-radius:8px;">
                  <h2 style="margin:0 0 8px;font-size:18px;">{index}. {html.escape(paper.title)}</h2>
                  <p style="margin:4px 0;color:#555;">作者：{html.escape(_format_authors(paper.authors))}</p>
                  <p style="margin:4px 0;color:#555;">发布：{paper.published.astimezone(ZoneInfo(cfg.timezone)).strftime('%Y-%m-%d')}</p>
                  <p style="margin:8px 0;"><a href="{html.escape(paper.arxiv_url)}">arXiv 论文页</a> ·
                  <a href="{html.escape(paper.pdf_url)}">PDF</a></p>
                  <p style="margin:8px 0;"><strong>核心介绍：</strong>{html.escape(summary)}</p>
                  {gh_html}
                </div>
                """
            )
        body = "".join(sections)

    return f"""
    <html>
      <body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:#111;max-width:760px;">
        <h1 style="font-size:22px;">🎵 人声/伴奏分离 · 每日论文推送</h1>
        <p style="color:#666;">{date_str} · 共 {len(papers)} 篇新论文</p>
        {body}
        <hr style="border:none;border-top:1px solid #eee;margin:24px 0;" />
        <p style="color:#999;font-size:12px;">由 Music Source Separation Training 项目的 paper_digest 自动推送</p>
      </body>
    </html>
    """


def send_digest_email(
    papers: list[tuple[Paper, str, list[str]]],
    cfg: PaperDigestConfig,
) -> None:
    if not cfg.recipient:
        raise ValueError("未配置收件邮箱 (PAPER_DIGEST_RECIPIENT 或 configs/paper_digest.yaml)")

    now = datetime.now(ZoneInfo(cfg.timezone))
    subject = f"【论文推送】人声/伴奏分离 · {now.strftime('%Y-%m-%d')} ({len(papers)} 篇)"
    html_content = build_html_email(papers, cfg)

    if cfg.auth_method == "graph":
        from scripts.paper_digest.graph_sender import get_graph_access_token, send_mail_via_graph

        access_token = get_graph_access_token(cfg)
        send_mail_via_graph(
            access_token=access_token,
            recipient=cfg.recipient,
            subject=subject,
            html_body=html_content,
        )
        return

    if not cfg.smtp_host or not cfg.smtp_user or not cfg.smtp_password:
        raise ValueError("未配置 SMTP (host/user/password)，或改用 auth_method: graph")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"{cfg.sender_name} <{cfg.smtp_user}>"
    msg["To"] = cfg.recipient
    msg.attach(MIMEText(html_content, "html", "utf-8"))

    with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=60) as server:
        if cfg.smtp_use_tls:
            server.starttls()
        server.login(cfg.smtp_user, cfg.smtp_password)
        server.sendmail(cfg.smtp_user, [cfg.recipient], msg.as_string())
