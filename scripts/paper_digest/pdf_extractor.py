from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
import requests

logger = logging.getLogger(__name__)

ARCH_KEYWORDS = re.compile(
    r"architecture|framework|overview|pipeline|block diagram|flowchart|"
    r"proposed method|schematic|system overview|model overview|"
    r"illustration of (?:the )?(?:proposed|our)|network architecture",
    re.IGNORECASE,
)
RESULT_KEYWORDS = re.compile(
    r"result|comparison|ablation|evaluation|experiment|performance|"
    r"benchmark|quantitative|score|sdr|pesq|stoi|wer|mos|"
    r"accuracy|curve|plot",
    re.IGNORECASE,
)


@dataclass
class ExtractedFigure:
    caption: str
    image_bytes: bytes
    mime_type: str = "image/png"
    kind: str = "other"  # architecture | result | other
    page: int = 0


@dataclass
class PdfExtract:
    full_text: str = ""
    abstract_text: str = ""
    intro_text: str = ""
    method_text: str = ""
    result_text: str = ""
    architecture_figures: list[ExtractedFigure] = field(default_factory=list)
    result_figures: list[ExtractedFigure] = field(default_factory=list)
    result_tables: list[str] = field(default_factory=list)


def download_arxiv_pdf(arxiv_id: str, cache_dir: Path, timeout: int = 90) -> Path | None:
    stable_id = arxiv_id.split("v")[0]
    cache_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = cache_dir / f"{stable_id}.pdf"
    if pdf_path.exists() and pdf_path.stat().st_size > 1000:
        return pdf_path

    url = f"https://arxiv.org/pdf/{stable_id}.pdf"
    try:
        response = requests.get(url, timeout=timeout, headers={"User-Agent": "paper-digest/1.0"})
        response.raise_for_status()
        pdf_path.write_bytes(response.content)
        return pdf_path
    except requests.RequestException as exc:
        logger.warning("下载 PDF 失败 %s: %s", arxiv_id, exc)
        return None


def _section_slice(text: str, start_pat: str, end_pats: list[str], max_chars: int = 2500) -> str:
    match = re.search(start_pat, text, flags=re.IGNORECASE | re.MULTILINE)
    if not match:
        return ""
    start = match.end()
    end = len(text)
    for end_pat in end_pats:
        end_match = re.search(end_pat, text[start:], flags=re.IGNORECASE | re.MULTILINE)
        if end_match:
            end = start + end_match.start()
            break
    return text[start:end].strip()[:max_chars]


def _extract_sections(text: str) -> tuple[str, str, str, str]:
    abstract = _section_slice(
        text,
        r"(?:^|\n)\s*abstract\s*\n",
        [r"(?:^|\n)\s*(?:1\.?\s+)?introduction\b", r"(?:^|\n)\s*keywords?\b"],
        max_chars=2000,
    )
    intro = _section_slice(
        text,
        r"(?:^|\n)\s*(?:1\.?\s+)?introduction\b",
        [r"(?:^|\n)\s*2\.?\s+", r"(?:^|\n)\s*(?:related work|method|approach|proposed)\b"],
        max_chars=2200,
    )
    method = _section_slice(
        text,
        r"(?:^|\n)\s*(?:\d\.?\s+)?(?:method|approach|proposed method|model|architecture)\b",
        [r"(?:^|\n)\s*(?:\d\.?\s+)?(?:experiment|evaluation|result|training)\b"],
        max_chars=2500,
    )
    result = _section_slice(
        text,
        r"(?:^|\n)\s*(?:\d\.?\s+)?(?:experiment|experimental results?|results?|evaluation)\b",
        [r"(?:^|\n)\s*(?:\d\.?\s+)?(?:conclusion|discussion|related work|acknowledg)\b"],
        max_chars=3500,
    )
    return abstract, intro, method, result


def _extract_tables_text(text: str, max_tables: int = 3) -> list[str]:
    tables: list[str] = []
    pattern = re.compile(
        r"(Table\s+\d+[.:][^\n]{0,200}\n(?:[^\n]*\n){1,22})",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        block = re.sub(r"[ \t]+", " ", match.group(1)).strip()
        if len(block) < 40:
            continue
        tables.append(block[:1500])
        if len(tables) >= max_tables:
            break
    return tables


def _classify_caption(caption: str) -> str:
    # Prefer result classification when captions mention evaluation/metrics
    if RESULT_KEYWORDS.search(caption):
        return "result"
    if ARCH_KEYWORDS.search(caption):
        return "architecture"
    return "other"


def _find_caption_blocks(page: fitz.Page, kind: str = "figure") -> list[tuple[fitz.Rect, str]]:
    if kind == "table":
        pattern = r"^Table\s*\d+"
    else:
        pattern = r"^(Figure|Fig\.?)\s*\d+"
    captions: list[tuple[fitz.Rect, str]] = []
    for block in page.get_text("blocks"):
        x0, y0, x1, y1, text, *_ = block
        clean = re.sub(r"\s+", " ", str(text)).strip()
        if re.match(pattern, clean, flags=re.IGNORECASE):
            captions.append((fitz.Rect(x0, y0, x1, y1), clean[:320]))
    return captions


def _compress_png(image_bytes: bytes, max_width: int = 900) -> bytes:
    """Downscale large screenshots so 10–15 papers fit in one email."""
    try:
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if image.width > max_width:
            ratio = max_width / float(image.width)
            image = image.resize((max_width, max(1, int(image.height * ratio))), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=82, optimize=True)
        return buf.getvalue()
    except Exception:
        return image_bytes


def _render_region_above_caption(
    page: fitz.Page,
    caption_rect: fitz.Rect,
    caption: str,
    kind: str,
    upward: float = 340,
) -> ExtractedFigure | None:
    """Render region above caption as image to capture vector figures/tables."""
    page_rect = page.rect
    top = max(page_rect.y0 + 15, caption_rect.y0 - upward)
    if caption_rect.y0 > page_rect.height * 0.45:
        top = max(page_rect.y0 + 15, caption_rect.y0 - upward - 80)
    bottom = min(page_rect.y1 - 5, caption_rect.y1 + 6)
    clip = fitz.Rect(page_rect.x0 + 16, top, page_rect.x1 - 16, bottom)
    if clip.height < 50 or clip.width < 80:
        return None
    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), clip=clip, alpha=False)
        image_bytes = _compress_png(pix.tobytes("png"))
    except Exception:
        return None
    if len(image_bytes) < 2500:
        return None
    return ExtractedFigure(
        caption=caption,
        image_bytes=image_bytes,
        mime_type="image/jpeg",
        kind=kind,
        page=page.number + 1,
    )


def _extract_figures(doc: fitz.Document, max_figures: int = 10) -> list[ExtractedFigure]:
    figures: list[ExtractedFigure] = []
    for page_index in range(min(len(doc), 14)):
        page = doc[page_index]
        for caption_rect, caption in _find_caption_blocks(page, "figure"):
            kind = _classify_caption(caption)
            fig = _render_region_above_caption(page, caption_rect, caption, kind=kind, upward=340)
            if fig:
                figures.append(fig)
            if len(figures) >= max_figures:
                return figures
    return figures


def _extract_table_images(doc: fitz.Document, max_tables: int = 3) -> list[ExtractedFigure]:
    tables: list[ExtractedFigure] = []
    for page_index in range(min(len(doc), 16)):
        page = doc[page_index]
        for caption_rect, caption in _find_caption_blocks(page, "table"):
            fig = _render_region_above_caption(page, caption_rect, caption, kind="result", upward=280)
            if fig:
                tables.append(fig)
            if len(tables) >= max_tables:
                return tables
    return tables


def extract_from_pdf(pdf_path: Path) -> PdfExtract:
    extract = PdfExtract()
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        logger.warning("打开 PDF 失败 %s: %s", pdf_path, exc)
        return extract

    try:
        texts = [page.get_text("text") for page in doc]
        extract.full_text = "\n".join(texts)
        extract.abstract_text, extract.intro_text, extract.method_text, extract.result_text = _extract_sections(
            extract.full_text
        )
        extract.result_tables = _extract_tables_text(extract.full_text)

        figures = _extract_figures(doc)
        table_images = _extract_table_images(doc)

        for fig in figures:
            if fig.kind == "architecture" and len(extract.architecture_figures) < 1:
                extract.architecture_figures.append(fig)
            elif fig.kind == "result" and len(extract.result_figures) < 1:
                extract.result_figures.append(fig)

        # Prefer rendered original tables as experimental results (up to 2)
        for fig in reversed(table_images):
            if len(extract.result_figures) < 2:
                extract.result_figures.insert(0, fig)

        if not extract.architecture_figures:
            for fig in figures:
                if fig.page <= 5:
                    extract.architecture_figures.append(fig)
                    break
        if not extract.result_figures:
            for fig in figures:
                if fig not in extract.architecture_figures:
                    extract.result_figures.append(fig)
                if len(extract.result_figures) >= 2:
                    break
    finally:
        doc.close()

    return extract
