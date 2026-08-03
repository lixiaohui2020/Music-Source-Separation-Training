from __future__ import annotations

import html
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from urllib.parse import quote
from zoneinfo import ZoneInfo

from scripts.paper_digest.arxiv_client import Paper
from scripts.paper_digest.config import PaperDigestConfig
from scripts.paper_digest.email_renderer import render_analysis_sections
from scripts.paper_digest.paper_analyzer import PaperAnalysis


def _format_authors(authors: list[str], max_count: int = 5) -> str:
    if not authors:
        return "未知"
    if len(authors) <= max_count:
        return ", ".join(authors)
    return ", ".join(authors[:max_count]) + f" 等 {len(authors)} 人"


def _mendeley_links(paper: Paper) -> tuple[str, str, str]:
    """Return (open_arxiv, bibtex, mendeley_import) URLs for one paper."""
    stable_id = paper.stable_id
    arxiv_url = paper.arxiv_url or f"https://arxiv.org/abs/{stable_id}"
    bibtex_url = f"https://arxiv.org/bibtex/{stable_id}"
    mendeley_import = f"https://www.mendeley.com/import/?url={quote(arxiv_url, safe='')}"
    return arxiv_url, bibtex_url, mendeley_import


def _render_mendeley_actions(paper: Paper) -> str:
    arxiv_url, bibtex_url, mendeley_import = _mendeley_links(paper)
    btn = (
        "display:inline-block;margin:4px 8px 4px 0;padding:8px 12px;"
        "border-radius:6px;text-decoration:none;font-size:13px;font-weight:600;"
    )
    return f"""
    <div style="margin:12px 0;padding:12px;background:#eef2ff;border-radius:8px;">
      <p style="margin:0 0 8px;font-weight:600;color:#111;">📚 添加到 Mendeley</p>
      <p style="margin:0 0 10px;color:#4b5563;font-size:13px;line-height:1.5;">
        推荐：先点「打开 arXiv」，再用浏览器插件 <b>Mendeley Web Importer</b> 一键入库（含 PDF）。
      </p>
      <a href="{html.escape(arxiv_url)}" style="{btn}background:#4f46e5;color:#fff;">打开 arXiv</a>
      <a href="{html.escape(mendeley_import)}" style="{btn}background:#0f766e;color:#fff;">Mendeley 导入页</a>
      <a href="{html.escape(bibtex_url)}" style="{btn}background:#ffffff;color:#111;border:1px solid #cbd5e1;">下载 BibTeX</a>
    </div>
    """


def _render_topic_tags(topics: list[str]) -> str:
    if not topics:
        return ""
    tags = "".join(
        f'<span style="display:inline-block;margin:2px 4px 2px 0;padding:2px 8px;'
        f'background:#f3f4f6;border-radius:12px;font-size:12px;color:#4b5563;">'
        f"{html.escape(t)}</span>"
        for t in topics
    )
    return f'<p style="margin:6px 0;">{tags}</p>'


def _format_multiline(text: str) -> str:
    return "<br/>".join(html.escape(line) for line in text.split("\n") if line.strip())


def build_html_email(
    papers: list[tuple[Paper, PaperAnalysis, list[str]]],
    cfg: PaperDigestConfig,
) -> str:
    now = datetime.now(ZoneInfo(cfg.timezone))
    date_str = now.strftime("%Y年%m月%d日")

    if not papers:
        body = "<p>今日暂无新的 AI 音频相关论文。</p>"
    else:
        sections = []
        for index, (paper, analysis, github_links) in enumerate(papers, start=1):
            gh_html = ""
            if github_links:
                gh_items = "".join(
                    f'<li><a href="{html.escape(url)}">{html.escape(url)}</a></li>'
                    for url in github_links
                )
                gh_html = f"<p><strong>参考 GitHub：</strong></p><ul>{gh_items}</ul>"
            else:
                gh_html = "<p><strong>参考 GitHub：</strong>暂未找到公开仓库</p>"

            rendered = render_analysis_sections(analysis)
            result_body = (
                rendered["result_tables_html"]
                + rendered["result_figures_html"]
                + rendered["highlights_html"]
            )
            if not result_body.strip():
                result_body = (
                    "<p style='color:#888;'>未从 PDF 提取到实验表格/结果图，请查看原文 PDF。</p>"
                )

            sections.append(
                f"""
                <div style="margin-bottom:28px;padding:18px;border:1px solid #e5e7eb;border-radius:10px;">
                  <h2 style="margin:0 0 8px;font-size:18px;">{index}. {html.escape(paper.title)}</h2>
                  {_render_topic_tags(analysis.topics)}
                  <p style="margin:4px 0;color:#555;">作者：{html.escape(_format_authors(paper.authors))}</p>
                  <p style="margin:4px 0;color:#555;">发布：{paper.published.astimezone(ZoneInfo(cfg.timezone)).strftime('%Y-%m-%d')}</p>
                  <p style="margin:8px 0;"><a href="{html.escape(paper.arxiv_url)}">arXiv 论文页</a> ·
                  <a href="{html.escape(paper.pdf_url)}">PDF</a></p>

                  {_render_mendeley_actions(paper)}

                  <div style="margin:12px 0;padding:12px;background:#f9fafb;border-radius:8px;">
                    <p style="margin:0 0 6px;font-weight:600;color:#111;">📌 核心介绍</p>
                    <p style="margin:0;color:#333;line-height:1.65;">{html.escape(analysis.brief_summary)}</p>
                  </div>

                  <div style="margin:12px 0;padding:12px;background:#fffbeb;border-left:4px solid #f59e0b;border-radius:4px;">
                    <p style="margin:0 0 6px;font-weight:600;color:#111;">💡 核心思路</p>
                    <p style="margin:0;color:#333;line-height:1.65;">{_format_multiline(analysis.core_idea)}</p>
                  </div>

                  <div style="margin:12px 0;padding:12px;background:#f0fdf4;border-radius:8px;">
                    <p style="margin:0 0 6px;font-weight:600;color:#111;">🔀 论文流程图 / 架构图（作者原图）</p>
                    {rendered["architecture_html"]}
                  </div>

                  <div style="margin:12px 0;padding:12px;background:#fef2f2;border-radius:8px;">
                    <p style="margin:0 0 6px;font-weight:600;color:#111;">📊 核心实验结果（论文原表/原图）</p>
                    {result_body}
                  </div>

                  {gh_html}
                </div>
                """
            )
        body = "".join(sections)

    return f"""
    <html>
      <body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:#111;max-width:860px;">
        <h1 style="font-size:22px;">🎧 {html.escape(cfg.digest_title)}</h1>
        <p style="color:#666;">{date_str} · 共 {len(papers)} 篇新论文 · 含作者原图与实验结果</p>
        {body}
        <hr style="border:none;border-top:1px solid #eee;margin:24px 0;" />
        <p style="color:#999;font-size:12px;">
          流程图与实验结果提取自 arXiv PDF 原文。感兴趣的论文可点「添加到 Mendeley」：
          先安装 <a href="https://www.mendeley.com/reference-management/web-importer">Mendeley Web Importer</a>，
          打开 arXiv 后点浏览器插件即可入库。
        </p>
      </body>
    </html>
    """


def send_digest_email(
    papers: list[tuple[Paper, PaperAnalysis, list[str]]],
    cfg: PaperDigestConfig,
) -> None:
    if not cfg.recipient:
        raise ValueError("未配置收件邮箱 (PAPER_DIGEST_RECIPIENT 或 configs/paper_digest.yaml)")

    now = datetime.now(ZoneInfo(cfg.timezone))
    subject = f"【论文推送】AI 音频 · {now.strftime('%Y-%m-%d')} ({len(papers)} 篇)"
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
