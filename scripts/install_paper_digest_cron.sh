#!/usr/bin/env bash
# Install a daily cron job to send vocal/accompaniment separation paper digest at 8:00 AM.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CRON_HOUR="${CRON_HOUR:-8}"
CRON_MINUTE="${CRON_MINUTE:-0}"
LOG_FILE="${REPO_ROOT}/data/paper_digest/cron.log"

mkdir -p "${REPO_ROOT}/data/paper_digest"

CRON_LINE="${CRON_MINUTE} ${CRON_HOUR} * * * cd ${REPO_ROOT} && ${PYTHON_BIN} -m scripts.paper_digest.main >> ${LOG_FILE} 2>&1"

# Remove any existing paper_digest cron entries, then add the new one
(crontab -l 2>/dev/null | grep -v "scripts.paper_digest.main" || true; echo "${CRON_LINE}") | crontab -

echo "✅ Cron job installed:"
echo "   ${CRON_LINE}"
echo ""
echo "请确保已配置 configs/paper_digest.yaml 或环境变量："
echo "   PAPER_DIGEST_RECIPIENT, PAPER_DIGEST_SMTP_HOST, PAPER_DIGEST_SMTP_USER, PAPER_DIGEST_SMTP_PASSWORD"
echo ""
echo "手动测试："
echo "   cd ${REPO_ROOT} && ${PYTHON_BIN} -m scripts.paper_digest.main --dry-run"
