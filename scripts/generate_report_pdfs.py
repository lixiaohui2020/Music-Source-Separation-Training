#!/usr/bin/env python3
"""Convert markdown reports to PDF using fpdf2 with Chinese font support."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from fpdf import FPDF

FONT_PATH = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
PAGE_W = 210
MARGIN = 15
CONTENT_W = PAGE_W - 2 * MARGIN


class ReportPDF(FPDF):
    def __init__(self, title: str = ""):
        super().__init__()
        self.doc_title = title
        self.add_font("zh", "", FONT_PATH)
        self.add_font("zh", "B", FONT_PATH)
        self.set_auto_page_break(auto=True, margin=18)

    def header(self):
        if self.page_no() > 1:
            self.set_font("zh", "", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 8, self.doc_title, align="R", new_x="LMARGIN", new_y="NEXT")
            self.line(MARGIN, self.get_y(), PAGE_W - MARGIN, self.get_y())
            self.ln(2)

    def footer(self):
        self.set_y(-12)
        self.set_font("zh", "", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"第 {self.page_no()} 页", align="C")


def write_wrapped(pdf: ReportPDF, text: str, size: int = 10, bold: bool = False, color=(30, 30, 30)):
    pdf.set_font("zh", "B" if bold else "", size)
    pdf.set_text_color(*color)
    pdf.multi_cell(CONTENT_W, size * 0.55, text)
    pdf.ln(1)


def write_code_block(pdf: ReportPDF, lines: list[str]):
    pdf.set_fill_color(15, 23, 42)
    pdf.set_text_color(226, 232, 240)
    pdf.set_font("zh", "", 8)
    y_start = pdf.get_y()
    text = "\n".join(lines)
    line_h = 4.2
    h = max(line_h * len(lines) + 6, 10)
    if pdf.get_y() + h > pdf.h - 20:
        pdf.add_page()
        y_start = pdf.get_y()
    pdf.rect(MARGIN, y_start, CONTENT_W, h, style="F")
    pdf.set_xy(MARGIN + 3, y_start + 3)
    pdf.multi_cell(CONTENT_W - 6, line_h, text)
    pdf.set_text_color(30, 30, 30)
    pdf.set_y(y_start + h + 2)


def write_table(pdf: ReportPDF, rows: list[list[str]]):
    if not rows:
        return
    col_count = max(len(r) for r in rows)
    col_w = CONTENT_W / col_count
    row_h = 7
    pdf.set_font("zh", "", 8)
    for i, row in enumerate(rows):
        if pdf.get_y() + row_h > pdf.h - 20:
            pdf.add_page()
        x = MARGIN
        y = pdf.get_y()
        if i == 0:
            pdf.set_fill_color(241, 245, 249)
            pdf.set_font("zh", "B", 8)
        else:
            pdf.set_fill_color(255, 255, 255)
            pdf.set_font("zh", "", 8)
        pdf.set_text_color(30, 30, 30)
        max_h = row_h
        for j in range(col_count):
            cell = row[j] if j < len(row) else ""
            pdf.set_xy(x + j * col_w, y)
            pdf.multi_cell(col_w, row_h, cell, border=1, fill=True, max_line_height=row_h)
            max_h = max(max_h, pdf.get_y() - y)
        pdf.set_y(y + max_h)
    pdf.ln(2)


def parse_table_block(lines: list[str]) -> list[list[str]]:
    rows = []
    for line in lines:
        if re.match(r"^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?$", line.strip()):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(cells)
    return rows


def md_to_pdf(md_path: Path, pdf_path: Path) -> None:
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    title = md_path.stem
    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip()
            break

    pdf = ReportPDF(title=title)
    pdf.add_page()

    i = 0
    in_code = False
    code_lines: list[str] = []
    table_lines: list[str] = []
    in_table = False

    while i < len(lines):
        line = lines[i]

        if line.strip().startswith("```"):
            if in_code:
                write_code_block(pdf, code_lines)
                code_lines = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue

        if in_code:
            code_lines.append(line)
            i += 1
            continue

        if line.strip().startswith("|"):
            if not in_table:
                in_table = True
                table_lines = []
            table_lines.append(line)
            i += 1
            continue
        elif in_table:
            write_table(pdf, parse_table_block(table_lines))
            table_lines = []
            in_table = False

        if line.startswith("# "):
            if pdf.get_y() > 30:
                pdf.add_page()
            write_wrapped(pdf, line[2:].strip(), size=18, bold=True, color=(30, 58, 95))
            pdf.ln(2)
        elif line.startswith("## "):
            pdf.ln(2)
            write_wrapped(pdf, line[3:].strip(), size=14, bold=True, color=(30, 64, 175))
        elif line.startswith("### "):
            write_wrapped(pdf, line[4:].strip(), size=12, bold=True, color=(51, 65, 85))
        elif line.startswith("#### "):
            write_wrapped(pdf, line[5:].strip(), size=11, bold=True, color=(71, 85, 105))
        elif line.strip() == "---":
            pdf.ln(2)
            y = pdf.get_y()
            pdf.set_draw_color(226, 232, 240)
            pdf.line(MARGIN, y, PAGE_W - MARGIN, y)
            pdf.ln(4)
        elif line.strip().startswith("> "):
            write_wrapped(pdf, line.strip()[2:], size=10, color=(30, 58, 95))
        elif re.match(r"^[-*]\s+", line.strip()):
            write_wrapped(pdf, "  • " + re.sub(r"^[-*]\s+", "", line.strip()), size=10)
        elif re.match(r"^\d+\.\s+", line.strip()):
            write_wrapped(pdf, "  " + line.strip(), size=10)
        elif line.strip():
            clean = re.sub(r"\*\*(.+?)\*\*", r"\1", line.strip())
            clean = re.sub(r"`(.+?)`", r"\1", clean)
            write_wrapped(pdf, clean, size=10)
        else:
            pdf.ln(2)

        i += 1

    if in_table and table_lines:
        write_table(pdf, parse_table_block(table_lines))
    if in_code and code_lines:
        write_code_block(pdf, code_lines)

    pdf.output(str(pdf_path))


def main() -> int:
    input_dir = Path("docs/algorithm-automation-reports")
    if not input_dir.is_dir():
        print(f"Directory not found: {input_dir}", file=sys.stderr)
        return 1

    pdf_dir = input_dir / "pdf"
    pdf_dir.mkdir(exist_ok=True)

    md_files = sorted(input_dir.glob("*.md"))
    if not md_files:
        print("No markdown files found.", file=sys.stderr)
        return 1

    for md_path in md_files:
        pdf_path = pdf_dir / f"{md_path.stem}.pdf"
        print(f"Converting: {md_path.name} -> {pdf_path.name}")
        md_to_pdf(md_path, pdf_path)

    print(f"\nDone. {len(md_files)} PDF(s) in {pdf_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
