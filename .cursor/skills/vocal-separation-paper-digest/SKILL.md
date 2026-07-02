---
name: vocal-separation-paper-digest
description: 每日抓取人声/伴奏分离相关 arXiv 论文，生成核心介绍与 GitHub 参考链接，并定时邮件推送。
alwaysApply: false
---

# 人声（伴奏）分离 · 每日论文推送

## 用途

自动监控 arXiv 上与 **人声分离 / 伴奏分离 / 音乐源分离（vocals）** 相关的新论文，整理后每天 8:00 发送到指定邮箱。

每篇论文包含：
- 标题、作者、arXiv 链接、发表日期
- **核心介绍**（从摘要提炼）
- **参考 GitHub 链接**（从摘要/备注提取，或按论文标题搜索）

## 目录结构

```
scripts/paper_digest/
  main.py           # 入口：抓取 → 去重 → 发信
  arxiv_client.py   # arXiv API 查询
  summarizer.py     # 摘要提炼
  github_finder.py  # GitHub 链接发现
  email_sender.py   # HTML 邮件
  storage.py        # 已推送论文去重
  config.py         # 配置加载
configs/paper_digest.example.yaml
scripts/install_paper_digest_cron.sh
```

## 快速开始

### 1. 安装依赖

```bash
pip install -e ".[paper-digest]"
```

### 2. 配置

复制示例配置并填写邮箱与 SMTP：

```bash
cp configs/paper_digest.example.yaml configs/paper_digest.yaml
# 编辑 configs/paper_digest.yaml
```

或使用环境变量（优先级高于配置文件）：

| 变量 | 说明 |
|------|------|
| `PAPER_DIGEST_RECIPIENT` | 收件邮箱 |
| `PAPER_DIGEST_SMTP_HOST` | SMTP 服务器 |
| `PAPER_DIGEST_SMTP_PORT` | 端口（默认 587） |
| `PAPER_DIGEST_SMTP_USER` | 发件账号 |
| `PAPER_DIGEST_SMTP_PASSWORD` | 密码/应用专用密码 |
| `PAPER_DIGEST_TIMEZONE` | 时区（默认 `Asia/Shanghai`） |

### 3. 手动试跑

```bash
python -m scripts.paper_digest.main --dry-run   # 仅打印，不发信
python -m scripts.paper_digest.main             # 抓取并发信
python -m scripts.paper_digest.main --force     # 忽略去重，重发今日内容
```

### 4. 安装每日 8:00 定时任务

```bash
bash scripts/install_paper_digest_cron.sh
```

默认 cron：`0 8 * * *`（`Asia/Shanghai` 时区 8:00）。

## 搜索范围

默认 arXiv 查询关键词（可在 `configs/paper_digest.yaml` 的 `search_queries` 中修改）：

- vocal separation / vocals separation
- accompaniment separation
- music source separation vocals
- singing voice separation
- stem separation vocals

分类限定：`cs.SD`（Sound）、`eess.AS`（Audio and Speech Processing）、`cs.LG`。

## 维护说明

- 已推送论文 ID 保存在 `data/paper_digest/sent_papers.json`
- 日志：`data/paper_digest/digest.log`
- 修改搜索词后无需清空历史；新论文会自动纳入
- 若 SMTP 使用 Gmail/QQ/163，需开启「应用专用密码」

## Agent 操作指引

当用户要求调整推送内容时：

1. 修改 `configs/paper_digest.yaml` 中的 `search_queries` 或 `max_papers_per_day`
2. 调整 `summarizer.py` 中的摘要提炼逻辑
3. 运行 `--dry-run` 验证输出
4. 确认 cron 时区与 `schedule_hour` 一致
