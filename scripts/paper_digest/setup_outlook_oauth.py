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
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not cfg.smtp_user:
        print("错误: 请在 configs/paper_digest.yaml 中配置 email.smtp_user", file=sys.stderr)
        sys.exit(1)
    if not cfg.graph_client_id:
        print(
            "\n请先在 Azure 注册应用并配置 client_id:\n"
            "1. 打开 https://portal.azure.com/#view/Microsoft_AAD_RegisteredApps/ApplicationsListBlade\n"
            "2. 新建注册 → 名称随意 → 账户类型选「个人 Microsoft 账户」\n"
            "3. 重定向 URI 选「公共客户端/本机」，填 http://localhost\n"
            "4. 注册后复制「应用程序(客户端) ID」到 configs/paper_digest.yaml 的 graph.client_id\n"
            "5. API 权限 → 添加 → Microsoft Graph → 委托权限 → Mail.Send → 授予同意\n"
            "6. 身份验证 → 高级设置 → 允许公共客户端流 → 是\n",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"将为账号 {cfg.smtp_user} 进行 OAuth 授权...")
    try:
        acquire_token_interactive(cfg)
        print("\n✅ 授权成功！token 已保存，可运行 python -m scripts.paper_digest.main 发送邮件。")
    except Exception as exc:
        print(f"\n❌ 授权失败: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
