#!/usr/bin/env bash
# port_sync.sh — WSL <-> Mac transport for the exp41-43 line.
#
# DISCIPLINE (doc 24 S8):
#   * Tracked files (code, datasets, results JSONs) move by GIT ONLY, via the
#     local bare hub. Never rsync them: git is the integrity check.
#   * Untracked artifacts that git cannot carry (per-config checkpoints, .pt
#     states, *.log which .gitignore excludes) move by this script.
#   * Every transfer is one-directional, checksum-verified, dry-run by default,
#     and NEVER deletes. Overwritten files are kept under .sync_backup/<date>.
#
# Usage:
#   ./port_sync.sh status                 # both trees, git + artifact state
#   ./port_sync.sh artifacts wsl2mac      # dry run
#   ./port_sync.sh artifacts wsl2mac --go # execute
#   ./port_sync.sh artifacts mac2wsl [--go]
#   ./port_sync.sh verify                 # hash-compare artifacts on both boxes
#   ./port_sync.sh backup                 # timestamped tgz of both repos on WSL
set -euo pipefail

WSL_HOST="waj@192.168.100.27"
WSL_PORT="2222"
WSL_DIR="/home/waj/discocat_arabic_v2"
MAC_DIR="/Users/wajahath/discocat_arabic_v2"
SSH="ssh -p ${WSL_PORT}"
STAMP="$(date +%Y-%m-%d_%H%M)"

# Untracked-but-valuable artifact classes. Extend here, not inline.
PATTERNS=(--include='exp43b_ckpt/' --include='exp43b_ckpt/*.json'
          --include='exp4*_ckpt/' --include='exp4*_ckpt/*.json'
          --include='*.pt' --include='exp4*.log' --include='*.DONE')

rsync_common=(-a --checksum --itemize-changes --human-readable
              --backup --backup-dir="../.sync_backup/${STAMP}"
              "${PATTERNS[@]}" --exclude='*')

cmd="${1:-status}"

case "$cmd" in
status)
  echo "=== MAC  ${MAC_DIR}"
  cd "$MAC_DIR"
  echo "  git:  $(git log --oneline -1)"
  echo "  diffs vs index: $(git status --porcelain | grep -vc '^??' || true)"
  echo "  artifacts: $(find . -name '*.pt' -o -name 'exp4*.log' -o -path './exp43b_ckpt/*.json' 2>/dev/null | wc -l | tr -d ' ') files"
  echo "=== WSL  ${WSL_DIR}"
  $SSH "$WSL_HOST" "cd ${WSL_DIR} && echo '  git:  '\$(git log --oneline -1) && \
    echo '  diffs vs index: '\$(git status --porcelain | grep -vc '^??' || true) && \
    echo '  artifacts: '\$(find . -name '*.pt' -o -name 'exp4*.log' -o -path './exp43b_ckpt/*.json' 2>/dev/null | wc -l) files && \
    echo '  running:  '\$(pgrep -af 'python3 -u exp4' | head -1)"
  ;;

artifacts)
  dir="${2:?direction required: wsl2mac | mac2wsl}"
  go="${3:-}"
  flags=("${rsync_common[@]}")
  [ "$go" = "--go" ] || flags+=(--dry-run)
  case "$dir" in
    wsl2mac) src="-e \"${SSH}\" ${WSL_HOST}:${WSL_DIR}/"; dst="${MAC_DIR}/"
             rsync "${flags[@]}" -e "$SSH" "${WSL_HOST}:${WSL_DIR}/" "${MAC_DIR}/" ;;
    mac2wsl) rsync "${flags[@]}" -e "$SSH" "${MAC_DIR}/" "${WSL_HOST}:${WSL_DIR}/" ;;
    *) echo "direction must be wsl2mac or mac2wsl"; exit 2 ;;
  esac
  [ "$go" = "--go" ] || echo "(DRY RUN — nothing written. Re-run with --go)"
  ;;

verify)
  echo "hashing artifacts on both boxes (this compares content, not mtimes)..."
  mac=$(cd "$MAC_DIR" && find . \( -name '*.pt' -o -name 'exp4*.log' -o -path './exp43b_ckpt/*.json' \) \
        -type f -print0 | sort -z | xargs -0 shasum -a 256 2>/dev/null | awk '{print $2" "$1}' | sort)
  wsl=$($SSH "$WSL_HOST" "cd ${WSL_DIR} && find . \( -name '*.pt' -o -name 'exp4*.log' -o -path './exp43b_ckpt/*.json' \) \
        -type f -print0 | sort -z | xargs -0 sha256sum 2>/dev/null | awk '{print \$2\" \"\$1}' | sort")
  diff <(echo "$mac") <(echo "$wsl") && echo "IDENTICAL on both boxes" || \
    echo "^^ '<' = Mac only/differs, '>' = WSL only/differs (a live run makes logs differ; that is expected)"
  ;;

backup)
  $SSH "$WSL_HOST" "mkdir -p /home/waj/qnlp_backups && cd /home/waj && \
    tar czf qnlp_backups/qnlp_${STAMP}.tgz --exclude=qiskit_lambeq_env --exclude=aravec \
    --exclude=__pycache__ --exclude='*:Zone.Identifier' discocat_arabic_v2 qnlp_private_docs && \
    sha256sum qnlp_backups/qnlp_${STAMP}.tgz"
  mkdir -p /Users/wajahath/qnlp_backups
  scp -P "$WSL_PORT" "${WSL_HOST}:/home/waj/qnlp_backups/qnlp_${STAMP}.tgz" /Users/wajahath/qnlp_backups/
  shasum -a 256 "/Users/wajahath/qnlp_backups/qnlp_${STAMP}.tgz"
  echo "compare the two hashes above — they must match"
  ;;

*) sed -n '2,25p' "$0"; exit 2 ;;
esac
