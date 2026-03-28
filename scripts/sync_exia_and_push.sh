#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET_DIR="${1:-${REPO_ROOT}/../Exia}"
COMMIT_MSG="${2:-chore: sync from lightning-core/Exia}"

"${SCRIPT_DIR}/sync_exia_to_standalone.sh" "${TARGET_DIR}"

if [[ ! -d "${TARGET_DIR}/.git" ]]; then
  echo "[sync-exia-push] target is not a git repository: ${TARGET_DIR}" >&2
  echo "[sync-exia-push] sync completed, but commit/push was skipped"
  exit 0
fi

pushd "${TARGET_DIR}" >/dev/null

git add .
if git diff --cached --quiet; then
  echo "[sync-exia-push] no changes to commit"
else
  git commit -m "${COMMIT_MSG}"
  git push origin main
  echo "[sync-exia-push] committed and pushed to origin/main"
fi

popd >/dev/null
