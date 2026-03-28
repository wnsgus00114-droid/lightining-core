#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOURCE_DIR="${REPO_ROOT}/Exia"
TARGET_DIR="${1:-${REPO_ROOT}/../Exia}"

if [[ ! -d "${SOURCE_DIR}" ]]; then
  echo "[sync-exia] source not found: ${SOURCE_DIR}" >&2
  exit 1
fi

mkdir -p "${TARGET_DIR}"

rsync -av --delete \
  --exclude ".git/" \
  --exclude "__pycache__/" \
  --exclude ".pytest_cache/" \
  --exclude "*.egg-info/" \
  "${SOURCE_DIR}/" "${TARGET_DIR}/"

echo "[sync-exia] synced ${SOURCE_DIR} -> ${TARGET_DIR}"
