#!/usr/bin/env bash
set -euo pipefail

REMOTE_NAME="origin"
DRY_RUN=0
CHECK_TARGET=1
FORCE=0

usage() {
  cat <<'EOF'
Usage:
  scripts/sync_remote_after_repo_rename.sh [--remote <name>] [--dry-run] [--no-check-target] [--force]

Options:
  --remote <name>     Remote name to update (default: origin)
  --dry-run           Print the new URL without changing git config
  --no-check-target   Skip git ls-remote availability check
  --force             Apply even if target repository is not reachable
  -h, --help          Show this help

Behavior:
  - Reads the current remote URL.
  - Rewrites "lightining-core" to "lightning-core".
  - By default, checks whether the target URL is reachable via git ls-remote.
  - If target is not reachable, it skips update unless --force is used.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote)
      if [[ $# -lt 2 ]]; then
        echo "error: --remote requires a value" >&2
        exit 2
      fi
      REMOTE_NAME="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --no-check-target)
      CHECK_TARGET=0
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "error: this script must run inside a git repository" >&2
  exit 1
fi

if ! git remote get-url "$REMOTE_NAME" >/dev/null 2>&1; then
  echo "error: remote '$REMOTE_NAME' does not exist" >&2
  exit 1
fi

CURRENT_URL="$(git remote get-url "$REMOTE_NAME")"
TARGET_URL="${CURRENT_URL/lightining-core/lightning-core}"

is_target_reachable() {
  local url="$1"
  git ls-remote --exit-code "$url" HEAD >/dev/null 2>&1
}

if [[ "$CURRENT_URL" == "$TARGET_URL" ]]; then
  echo "no change needed: '$REMOTE_NAME' already uses lightning-core"
  echo "current url: $CURRENT_URL"
  exit 0
fi

if [[ $CHECK_TARGET -eq 1 && $FORCE -eq 0 ]]; then
  if ! is_target_reachable "$TARGET_URL"; then
    echo "skip: target repository is not reachable yet"
    echo "target url: $TARGET_URL"
    echo "hint: run again after GitHub rename is complete, or use --force"
    exit 0
  fi
fi

if [[ $DRY_RUN -eq 1 ]]; then
  echo "dry-run: '$REMOTE_NAME' would be updated"
  echo "from: $CURRENT_URL"
  echo "to  : $TARGET_URL"
  exit 0
fi

git remote set-url "$REMOTE_NAME" "$TARGET_URL"

echo "updated remote '$REMOTE_NAME'"
echo "from: $CURRENT_URL"
echo "to  : $TARGET_URL"
echo

git remote -v
