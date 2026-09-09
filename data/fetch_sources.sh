#!/usr/bin/env bash
# Clone the upstream codebases used by text_image_dataset_builder.py
# into data/sources/<name> (or $OPTIZIP_SOURCES).
#
# Usage (from anywhere):
#   bash data/fetch_sources.sh            # all repos
#   bash data/fetch_sources.sh rust linux # subset
#   OPTIZIP_SOURCES=/data/code bash data/fetch_sources.sh
#
# Re-runnable: existing clones are fetched and the sparse cone is re-applied.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCES_ROOT="${OPTIZIP_SOURCES:-$SCRIPT_DIR/sources}"
mkdir -p "$SOURCES_ROOT"

log() { printf '==> %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# name|url|sparse_paths (space-separated; empty = full tree)
REPOS=(
  "firefox|https://github.com/mozilla/gecko-dev.git|mfbt xpcom dom layout js/src netwerk gfx widget ipc memory mozglue modules toolkit"
  "cpython|https://github.com/python/cpython.git|"
  "rust|https://github.com/rust-lang/rust.git|library compiler"
  "linux|https://github.com/torvalds/linux.git|include/linux include/asm-generic kernel/locking lib"
  "nasm|https://github.com/netwide-assembler/nasm.git|"
  "spring|https://github.com/spring-projects/spring-framework.git|"
  "roslyn|https://github.com/dotnet/roslyn.git|"
)

repo_field() {
  local entry="$1" field="$2"
  case "$field" in
    name)   echo "${entry%%|*}" ;;
    url)    echo "${entry}" | cut -d'|' -f2 ;;
    sparse) echo "${entry}" | cut -d'|' -f3- ;;
  esac
}

all_names() {
  local e
  for e in "${REPOS[@]}"; do
    repo_field "$e" name
  done
}

lookup() {
  local want="$1" e
  for e in "${REPOS[@]}"; do
    if [[ "$(repo_field "$e" name)" == "$want" ]]; then
      echo "$e"
      return 0
    fi
  done
  return 1
}

apply_sparse() {
  local dest="$1" sparse="$2"
  git -C "$dest" sparse-checkout init --cone >/dev/null
  if [[ -n "$sparse" ]]; then
    # shellcheck disable=SC2086
    git -C "$dest" sparse-checkout set $sparse
  else
    git -C "$dest" sparse-checkout disable >/dev/null || true
  fi
}

clone_or_update() {
  local name="$1" url="$2" sparse="$3"
  local dest="$SOURCES_ROOT/$name"

  if [[ -d "$dest/.git" ]]; then
    log "$name: fetch + update ($dest)"
    git -C "$dest" remote set-url origin "$url"
    git -C "$dest" fetch --filter=blob:none --prune origin
    apply_sparse "$dest" "$sparse"
    local branch
    branch="$(git -C "$dest" rev-parse --abbrev-ref origin/HEAD 2>/dev/null | sed 's#^origin/##')"
    branch="${branch:-HEAD}"
    git -C "$dest" checkout --force "origin/$branch" 2>/dev/null \
      || git -C "$dest" checkout --force FETCH_HEAD
  else
    log "$name: clone $url → $dest"
    rm -rf "$dest"
    if [[ -n "$sparse" ]]; then
      git clone --filter=blob:none --sparse --single-branch "$url" "$dest"
      apply_sparse "$dest" "$sparse"
      git -C "$dest" checkout
    else
      git clone --filter=blob:none --single-branch "$url" "$dest"
    fi
  fi

  log "$name: $(git -C "$dest" rev-parse --short HEAD)  $(du -sh "$dest" | cut -f1)"
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [name ...]

Repos: $(all_names | tr '\n' ' ')

Env:
  OPTIZIP_SOURCES   destination root (default: $SCRIPT_DIR/sources)
EOF
}

main() {
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
  fi

  command -v git >/dev/null || die "git is required"

  local names=("$@")
  if [[ ${#names[@]} -eq 0 ]]; then
    mapfile -t names < <(all_names)
  fi

  log "SOURCES_ROOT=$SOURCES_ROOT"

  local name entry url sparse
  for name in "${names[@]}"; do
    entry="$(lookup "$name")" || die "unknown repo: $name"
    url="$(repo_field "$entry" url)"
    sparse="$(repo_field "$entry" sparse)"
    clone_or_update "$name" "$url" "$sparse"
  done

  log "done"
}

main "$@"
