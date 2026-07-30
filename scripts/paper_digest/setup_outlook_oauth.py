"""One-time OAuth setup for Outlook personal accounts (Microsoft Graph)."""

from __future__ import annotations

import argparse
import sys

from scripts.paper_digest.config import load_config
from scripts.paper_digest.graph_sender import acquire_token_interactive


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Outlook 个人邮箱 OAuth 授权（Microsoft Graph API）"
    )
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--client-id", type=str, default=None, help="Azure 应用程序客户端 ID")
    parser.add_argument("--user-code", type=str, default=None, help="已获取的设备码（用于提示核对）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.client_id:
        cfg.graph_client_id = args.client_id.strip()
    if not cfg.smtp_user:
        print("错误: 请在 configs/paper_digest.yaml 中配置 email.smtp_user", file=sys.stderr)
        sys.exit(1)
    if not cfg.graph_client_id:
        print(
            "\n❌ 未配置 graph.client_id。Outlook 个人邮箱必须使用你自己注册的 Azure 应用。\n"
            "详细步骤见 docs/paper_digest_outlook_setup.md\n\n"
            "快速步骤:\n"
            "1. 打开 https://portal.azure.com/#view/Microsoft_AAD_RegisteredApps/ApplicationsListBlade\n"
            "2. 新建注册 → 账户类型选「个人 Microsoft 账户」\n"
            "3. 重定向 URI: 公共客户端/本机 → http://localhost\n"
            "4. API 权限 → Microsoft Graph → 委托权限 → Mail.Send\n"
            "5. 身份验证 → 允许公共客户端流 → 是\n"
            "6. 复制应用程序(客户端) ID，填入 configs/paper_digest.yaml 或:\n"
            "   python -m scripts.paper_digest.setup_outlook_oauth --client-id <你的客户端ID>\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # 微软第一方应用 ID 无法用于用户自行授权
    KNOWN_FIRST_PARTY_IDS = {
        "1950a258-227b-4e31-a9cf-717495945fc2",  # Azure PowerShell
        "04b07795-8ddb-461a-bbee-02f9e1bf7b46",  # Azure CLI
        "14d82eec-204b-4c3f-b7e5-296a38970923",  # Graph PowerShell
    }
    if cfg.graph_client_id.lower() in KNOWN_FIRST_PARTY_IDS:
        print(
            "\n❌ 不能使用 Microsoft 官方公共客户端 ID。\n"
            "请按 docs/paper_digest_outlook_setup.md 注册你自己的 Azure 应用。\n",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"将为账号 {cfg.smtp_user} 进行 OAuth 授权...")
    try:
        acquire_token_interactive(cfg, user_code_hint=args.user_code)
        print("\n✅ 授权成功！token 已保存，可运行 python -m scripts.paper_digest.main 发送邮件。")
    except Exception as exc:
        msg = str(exc)
        print(f"\n❌ 授权失败: {msg}", file=sys.stderr)
        if "first party application" in msg.lower():
            print(
                "\n原因: 使用了 Microsoft 官方应用 ID，个人用户无法授权。\n"
                "解决: 请注册你自己的 Azure 应用，见 docs/paper_digest_outlook_setup.md\n",
                file=sys.stderr,
            )
        sys.exit(1)


if __name__ == "__main__":
    main()
