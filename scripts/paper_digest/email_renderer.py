from __future__ import annotations

import html

from scripts.paper_digest.paper_analyzer import PaperAnalysis


def render_pipeline_flowchart(steps: list[str]) -> str:
    """Render an email-safe HTML flowchart from pipeline steps."""
    if not steps:
        return "<p style='color:#888;'>暂未从摘要中提取到明确流程步骤。</p>"

    boxes = []
    for index, step in enumerate(steps):
        boxes.append(
            f"""
            <td style="padding:4px 0;text-align:center;">
              <div style="display:inline-block;min-width:120px;max-width:160px;padding:10px 12px;
                          background:#eff6ff;border:1px solid #93c5fd;border-radius:8px;
                          font-size:13px;line-height:1.4;color:#1e3a5f;">
                <div style="font-size:11px;color:#3b82f6;margin-bottom:4px;">Step {index + 1}</div>
                {html.escape(step)}
              </div>
            </td>
            """
        )
        if index < len(steps) - 1:
            boxes.append(
                '<td style="padding:4px 8px;color:#6b7280;font-size:18px;vertical-align:middle;">→</td>'
            )

    return f"""
    <table role="presentation" cellpadding="0" cellspacing="0" style="margin:8px 0;border-collapse:collapse;">
      <tr>{''.join(boxes)}</tr>
    </table>
    """


def render_results_list(results: list[str]) -> str:
    if not results:
        return "<p style='color:#888;'>摘要中未明确报告量化实验结果，建议查看 PDF 全文。</p>"
    items = "".join(
        f"<li style='margin:4px 0;'>{html.escape(r)}</li>" for r in results
    )
    return f"<ul style='margin:6px 0;padding-left:20px;color:#333;'>{items}</ul>"
