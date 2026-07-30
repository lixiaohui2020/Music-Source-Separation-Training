from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from scripts.paper_digest.arxiv_client import deduplicate_papers, fetch_recent_papers
from scripts.paper_digest.config import load_config
from scripts.paper_digest.email_sender import build_html_email, send_digest_email
from scripts.paper_digest.github_finder import find_github_links
from scripts.paper_digest.storage import SentPaperStore
from scripts.paper_digest.summarizer import summarize_paper


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def run_digest(*, dry_run: bool = False, force: bool = False, config_path: str | None = None) -> int:
    cfg = load_config(config_path)
    setup_logging(cfg.log_path)

    logging.info("开始抓取 arXiv 论文...")
    store = SentPaperStore(cfg.sent_papers_path)
    is_first_run = not store._data.get("papers")
    effective_lookback = cfg.initial_lookback_days if is_first_run else cfg.lookback_days
    if is_first_run:
        logging.info("首次运行，回溯 %d 天", effective_lookback)

    papers = fetch_recent_papers(
        search_queries=cfg.search_queries,
        categories=cfg.categories,
        lookback_days=effective_lookback,
        max_results=cfg.max_papers_per_day * 5,
    )
    papers = deduplicate_papers(papers)
    logging.info("共获取 %d 篇候选论文", len(papers))

    selected = []
    for paper in papers:
        if force or not store.has_been_sent(paper.stable_id):
            selected.append(paper)
        if len(selected) >= cfg.max_papers_per_day:
            break

    digest_items: list[tuple] = []
    for paper in selected:
        summary = summarize_paper(paper)
        github_links = find_github_links(paper)
        digest_items.append((paper, summary, github_links))
        logging.info("选中: %s (%s)", paper.title, paper.arxiv_id)

    if dry_run:
        html_preview = build_html_email(digest_items, cfg)
        print(html_preview)
        logging.info("dry-run 模式，未发送邮件")
        return 0

    if not digest_items:
        logging.info("没有新论文，发送空日报")
    else:
        logging.info("准备发送 %d 篇论文", len(digest_items))

    send_digest_email(digest_items, cfg)

    for paper, _, _ in digest_items:
        store.mark_sent(paper.stable_id, paper.title)
    store.save()

    logging.info("邮件已发送至 %s", cfg.recipient)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="人声/伴奏分离每日论文推送")
    parser.add_argument("--dry-run", action="store_true", help="仅生成 HTML 预览，不发送邮件")
    parser.add_argument("--force", action="store_true", help="忽略去重记录，重新推送")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    args = parser.parse_args()

    try:
        code = run_digest(dry_run=args.dry_run, force=args.force, config_path=args.config)
    except Exception as exc:
        logging.exception("推送失败: %s", exc)
        sys.exit(1)
    sys.exit(code)


if __name__ == "__main__":
    main()
