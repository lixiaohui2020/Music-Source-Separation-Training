from __future__ import annotations

import base64
import html

from scripts.paper_digest.paper_analyzer import PaperAnalysis
from scripts.paper_digest.pdf_extractor import ExtractedFigure


def _img_data_uri(fig: ExtractedFigure) -> str:
    b64 = base64.b64encode(fig.image_bytes).decode("ascii")
    return f"data:{fig.mime_type};base64,{b64}"


def render_author_figures(figures: list[ExtractedFigure], title: str) -> str:
    if not figures:
        return (
            f"<p style='color:#888;margin:6px 0;'>未从 PDF 中提取到{html.escape(title)}，"
            "请点击上方 PDF 链接查看原文插图。</p>"
        )

    blocks = []
    for index, fig in enumerate(figures, start=1):
        caption = html.escape(fig.caption or f"{title} {index}")
        blocks.append(
            f"""
            <div style="margin:10px 0;text-align:center;">
              <img src="{_img_data_uri(fig)}" alt="{caption}"
                   style="max-width:100%;height:auto;border:1px solid #e5e7eb;border-radius:6px;" />
              <p style="margin:6px 0 0;font-size:12px;color:#555;">{caption}</p>
            </div>
            """
        )
    return "".join(blocks)


def render_result_tables(tables: list[str]) -> str:
    if not tables:
        return ""
    blocks = []
    for table in tables:
        escaped = html.escape(table)
        blocks.append(
            f"""
            <pre style="white-space:pre-wrap;word-break:break-word;background:#fff;
                        border:1px solid #fecaca;border-radius:6px;padding:10px;
                        font-size:12px;line-height:1.45;color:#111;margin:8px 0;">{escaped}</pre>
            """
        )
    return "".join(blocks)


def render_highlights(items: list[str]) -> str:
    if not items:
        return ""
    lis = "".join(f"<li style='margin:4px 0;'>{html.escape(i)}</li>" for i in items)
    return f"<ul style='margin:6px 0;padding-left:20px;'>{lis}</ul>"


def render_analysis_sections(analysis: PaperAnalysis) -> dict[str, str]:
    # Prefer original table/result images; keep text tables only as backup
    result_figures_html = render_author_figures(analysis.result_figures, "实验结果图/表")
    result_tables_html = ""
    if not analysis.result_figures:
        result_tables_html = render_result_tables(analysis.result_tables)
    return {
        "architecture_html": render_author_figures(analysis.architecture_figures, "作者流程图/架构图"),
        "result_figures_html": result_figures_html,
        "result_tables_html": result_tables_html,
        "highlights_html": render_highlights(analysis.experiment_highlights),
    }
