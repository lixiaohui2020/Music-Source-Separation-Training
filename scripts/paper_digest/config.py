from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_SEARCH_QUERIES = [
    # 降噪 / 增强
    'all:"speech enhancement" OR all:"audio denoising" OR all:"speech denoising"',
    'all:"noise suppression" OR all:"audio restoration"',
    # 去回声 / 去混响
    'all:"echo cancellation" OR all:dereverberation OR all:"acoustic echo"',
    # 源分离 / 人声分离
    'all:"source separation" OR all:"vocal separation" OR all:"music source separation"',
    'all:"singing voice separation" OR all:"stem separation"',
    # 语音识别 / 转录
    'all:"speech recognition" OR all:"automatic speech recognition" OR all:ASR',
    'all:"speech transcription" OR all:"speech-to-text" OR all:whisper',
    # 语音交互 / 对话
    'all:"spoken dialogue" OR all:"voice interaction" OR all:"spoken language model"',
    'all:"speech understanding" OR all:"audio language model"',
    # 语音合成 / 生成
    'all:"text-to-speech" OR all:"speech synthesis" OR all:"neural vocoder"',
    'all:"audio generation" OR all:"sound generation" OR all:"music generation"',
    # 说话人 / 声纹
    'all:"speaker diarization" OR all:"voice conversion" OR all:"speaker verification"',
    # 音频基础模型
    'all:"audio large language model" OR all:"audio LLM" OR all:"speech foundation model"',
]

DEFAULT_CATEGORIES = ["cs.SD", "eess.AS", "cs.CL", "cs.LG", "cs.AI"]


@dataclass
class PaperDigestConfig:
    auth_method: str = "smtp"  # smtp | graph
    recipient: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_use_tls: bool = True
    sender_name: str = "AI Audio Paper Digest"
    digest_title: str = "AI 音频 · 每日论文推送"
    graph_client_id: str = ""
    graph_authority: str = "https://login.microsoftonline.com/consumers"
    timezone: str = "Asia/Shanghai"
    schedule_hour: int = 8
    max_papers_per_day: int = 15
    lookback_days: int = 7
    initial_lookback_days: int = 30
    search_queries: list[str] = field(default_factory=lambda: list(DEFAULT_SEARCH_QUERIES))
    categories: list[str] = field(default_factory=lambda: list(DEFAULT_CATEGORIES))
    data_dir: Path = field(default_factory=lambda: Path("data/paper_digest"))
    config_path: Path | None = None

    @property
    def sent_papers_path(self) -> Path:
        return self.data_dir / "sent_papers.json"

    @property
    def log_path(self) -> Path:
        return self.data_dir / "digest.log"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def load_config(config_path: str | Path | None = None) -> PaperDigestConfig:
    repo_root = Path(__file__).resolve().parents[2]
    candidates = []
    if config_path:
        candidates.append(Path(config_path))
    candidates.extend(
        [
            repo_root / "configs" / "paper_digest.yaml",
            repo_root / "configs" / "paper_digest.local.yaml",
        ]
    )

    raw: dict = {}
    resolved_path: Path | None = None
    for candidate in candidates:
        if candidate.is_file():
            with candidate.open(encoding="utf-8") as handle:
                raw = yaml.safe_load(handle) or {}
            resolved_path = candidate
            break

    cfg = PaperDigestConfig()
    if resolved_path:
        cfg.config_path = resolved_path

    email_cfg = raw.get("email", {})
    schedule_cfg = raw.get("schedule", {})
    search_cfg = raw.get("search", {})
    paths_cfg = raw.get("paths", {})

    cfg.recipient = _env("PAPER_DIGEST_RECIPIENT", email_cfg.get("recipient", cfg.recipient))
    cfg.auth_method = _env("PAPER_DIGEST_AUTH_METHOD", email_cfg.get("auth_method", cfg.auth_method)).lower()
    cfg.smtp_host = _env("PAPER_DIGEST_SMTP_HOST", email_cfg.get("smtp_host", cfg.smtp_host))
    cfg.smtp_port = int(_env("PAPER_DIGEST_SMTP_PORT", str(email_cfg.get("smtp_port", cfg.smtp_port))))
    cfg.smtp_user = _env("PAPER_DIGEST_SMTP_USER", email_cfg.get("smtp_user", cfg.smtp_user))
    cfg.smtp_password = _env("PAPER_DIGEST_SMTP_PASSWORD", email_cfg.get("smtp_password", cfg.smtp_password))
    cfg.smtp_use_tls = bool(email_cfg.get("smtp_use_tls", cfg.smtp_use_tls))
    cfg.sender_name = email_cfg.get("sender_name", cfg.sender_name)
    cfg.digest_title = email_cfg.get("digest_title", cfg.digest_title)

    graph_cfg = email_cfg.get("graph", {})
    cfg.graph_client_id = _env("PAPER_DIGEST_GRAPH_CLIENT_ID", graph_cfg.get("client_id", cfg.graph_client_id))
    cfg.graph_authority = graph_cfg.get("authority", cfg.graph_authority)

    cfg.timezone = _env("PAPER_DIGEST_TIMEZONE", schedule_cfg.get("timezone", cfg.timezone))
    cfg.schedule_hour = int(schedule_cfg.get("hour", cfg.schedule_hour))

    cfg.max_papers_per_day = int(search_cfg.get("max_papers_per_day", cfg.max_papers_per_day))
    cfg.lookback_days = int(search_cfg.get("lookback_days", cfg.lookback_days))
    cfg.initial_lookback_days = int(search_cfg.get("initial_lookback_days", cfg.initial_lookback_days))
    if search_cfg.get("queries"):
        cfg.search_queries = list(search_cfg["queries"])
    if search_cfg.get("categories"):
        cfg.categories = list(search_cfg["categories"])

    data_dir = paths_cfg.get("data_dir")
    if data_dir:
        cfg.data_dir = Path(data_dir)
        if not cfg.data_dir.is_absolute():
            cfg.data_dir = repo_root / cfg.data_dir

    return cfg
