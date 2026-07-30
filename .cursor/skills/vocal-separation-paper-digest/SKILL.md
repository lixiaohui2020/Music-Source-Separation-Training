---
name: ai-audio-paper-digest
description: 每日抓取 AI 音频相关 arXiv 论文（降噪、去回声、人声分离、语音转录、语音交互等），生成核心介绍与 GitHub 链接，并定时邮件推送。
alwaysApply: false
---

# AI 音频 · 每日论文推送

## 覆盖领域

- AI 降噪 / 语音增强
- 去回声 / 去混响
- 人声分离 / 音乐源分离
- 语音识别 / 语音转录（ASR）
- 语音交互 / 口语对话
- 语音合成 / 音频生成
- 说话人识别 / 声纹 / 语音转换
- 音频大模型 / 多模态语音

## 快速开始

```bash
pip install -e ".[paper-digest]"
cp configs/paper_digest.example.yaml configs/paper_digest.yaml
python -m scripts.paper_digest.main --dry-run
python -m scripts.paper_digest.main
bash scripts/install_paper_digest_cron.sh   # 本机 cron
```

## GitHub Actions 定时推送

工作流：`.github/workflows/paper_digest.yml`（每天 8:00 北京时间）

Fork 仓库需在 **Actions** 页手动启用 workflow，并配置 Secret：

- `PAPER_DIGEST_SMTP_PASSWORD`：QQ 邮箱 SMTP 授权码

## 配置要点

- 收件邮箱：`email.recipient`
- 每日上限：`search.max_papers_per_day`（默认 15）
- 搜索词：`search.queries`（见 `configs/paper_digest.example.yaml`）
- 分类：`cs.SD`, `eess.AS`, `cs.CL`, `cs.LG`, `cs.AI`

## 维护

- 去重记录：`data/paper_digest/sent_papers.json`
- 日志：`data/paper_digest/digest.log`
- Outlook 发信见 `docs/paper_digest_outlook_setup.md`
