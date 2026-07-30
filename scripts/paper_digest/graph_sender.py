from __future__ import annotations

import logging
from pathlib import Path

import requests

from scripts.paper_digest.config import PaperDigestConfig

logger = logging.getLogger(__name__)

GRAPH_SEND_MAIL_URL = "https://graph.microsoft.com/v1.0/me/sendMail"
DEFAULT_SCOPES = ["https://graph.microsoft.com/Mail.Send"]


def _token_cache_path(cfg: PaperDigestConfig) -> Path:
    return cfg.data_dir / "ms_token_cache.json"


def _build_msal_app(cfg: PaperDigestConfig):
    try:
        import msal
    except ImportError as exc:
        raise ImportError("请安装 msal: pip install -e '.[paper-digest]'") from exc

    cache_path = _token_cache_path(cfg)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    token_cache = msal.SerializableTokenCache()
    if cache_path.exists():
        token_cache.deserialize(cache_path.read_text(encoding="utf-8"))

    app = msal.PublicClientApplication(
        cfg.graph_client_id,
        authority=cfg.graph_authority,
        token_cache=token_cache,
    )
    return app, token_cache, cache_path


def _save_cache(token_cache, cache_path: Path) -> None:
    if token_cache.has_state_changed:
        cache_path.write_text(token_cache.serialize(), encoding="utf-8")


def acquire_token_interactive(cfg: PaperDigestConfig, *, user_code_hint: str | None = None) -> str:
    """One-time device-code authorization for Outlook personal accounts."""
    if not cfg.graph_client_id:
        raise ValueError(
            "未配置 graph.client_id。请在 Azure 注册应用后填入，"
            "或设置环境变量 PAPER_DIGEST_GRAPH_CLIENT_ID"
        )

    app, token_cache, cache_path = _build_msal_app(cfg)
    accounts = app.get_accounts(username=cfg.smtp_user or None)

    if accounts:
        result = app.acquire_token_silent(DEFAULT_SCOPES, account=accounts[0])
        if result and "access_token" in result:
            _save_cache(token_cache, cache_path)
            return result["access_token"]

    flow = app.initiate_device_flow(scopes=DEFAULT_SCOPES)
    if "user_code" not in flow:
        raise RuntimeError(f"无法启动设备码授权: {flow}")

    print("\n" + "=" * 60)
    print(flow["message"])
    if user_code_hint and flow.get("user_code") != user_code_hint:
        print(f"\n注意: 你提供的设备码 {user_code_hint} 与本次会话不匹配。")
        print(f"请使用本次设备码: {flow.get('user_code')}")
        print("（设备码每次启动授权都会变化，且 15 分钟内有效）")
    print("=" * 60 + "\n")

    result = app.acquire_token_by_device_flow(flow)
    if "access_token" not in result:
        raise RuntimeError(result.get("error_description") or "OAuth 授权失败")

    _save_cache(token_cache, cache_path)
    logger.info("Outlook OAuth 授权成功")
    return result["access_token"]


def get_graph_access_token(cfg: PaperDigestConfig) -> str:
    if not cfg.graph_client_id:
        raise ValueError("未配置 graph.client_id")

    app, token_cache, cache_path = _build_msal_app(cfg)
    accounts = app.get_accounts(username=cfg.smtp_user or None)

    if accounts:
        result = app.acquire_token_silent(DEFAULT_SCOPES, account=accounts[0])
        if result and "access_token" in result:
            _save_cache(token_cache, cache_path)
            return result["access_token"]

    raise RuntimeError(
        "Outlook OAuth token 已过期或未授权。请运行:\n"
        "  python -m scripts.paper_digest.setup_outlook_oauth"
    )


def send_mail_via_graph(
    *,
    access_token: str,
    recipient: str,
    subject: str,
    html_body: str,
) -> None:
    response = requests.post(
        GRAPH_SEND_MAIL_URL,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        json={
            "message": {
                "subject": subject,
                "body": {"contentType": "HTML", "content": html_body},
                "toRecipients": [{"emailAddress": {"address": recipient}}],
            },
            "saveToSentItems": True,
        },
        timeout=60,
    )
    if response.status_code not in (200, 202):
        raise RuntimeError(f"Graph API 发信失败 ({response.status_code}): {response.text}")
