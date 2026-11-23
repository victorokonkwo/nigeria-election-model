#!/usr/bin/env bash
# Reproduce the pre-registered 2027 forecast(s) from a clean checkout of a commit.
#
# Checks out COMMIT (default: HEAD) into a temporary git worktree, copies only
# the local data inputs that are not in git (data/interim, data/raw), rebuilds
# data/processed/panel.parquet with the committed code, reruns the forecast
# script(s) present in that commit, and compares SHA-256 hashes with the
# issued files. Writes outputs/exploratory/reproduction_check.json.
#
# Usage: scripts/check_reproduction.sh [COMMIT]
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
COMMIT="${1:-HEAD}"
SHA="$(git -C "$ROOT" rev-parse "$COMMIT")"
WT="$(mktemp -d)/wt"
PY="$ROOT/.venv/bin/python"
OUT="$ROOT/outputs/exploratory/reproduction_check.json"

git -C "$ROOT" worktree add --detach "$WT" "$SHA" >/dev/null
trap 'git -C "$ROOT" worktree remove --force "$WT" >/dev/null 2>&1 || true' EXIT
mkdir -p "$WT/data"
cp -R "$ROOT/data/interim" "$WT/data/"
[ -d "$ROOT/data/raw" ] && cp -R "$ROOT/data/raw" "$WT/data/"

cd "$WT"
export PYTHONPATH="$WT/src:$WT/scripts"
"$PY" scripts/build_panel.py >/dev/null 2>&1

hash() { shasum -a 256 "$1" | cut -d' ' -f1; }
results=()
add() { results+=("\"$1\": {\"reproduced\": \"$2\", \"issued\": \"$3\", \"identical\": $( [ "$2" = "$3" ] && echo true || echo false )}"); }

add "data/processed/panel.parquet" "$(hash data/processed/panel.parquet)" "$(hash "$ROOT/data/processed/panel.parquet")"
for variant in 2027 2027_v3; do
  script="scripts/forecast_${variant/2027/2027}.py"
  [ "$variant" = "2027" ] && script="scripts/forecast_2027.py"
  [ "$variant" = "2027_v3" ] && script="scripts/forecast_2027_v3.py"
  [ -f "$script" ] || continue
  "$PY" "$script" >/dev/null 2>&1
  for f in states.csv conditional.csv; do
    add "outputs/preregistered/$variant/$f" \
      "$(hash "outputs/preregistered/$variant/$f")" \
      "$(hash "$ROOT/outputs/preregistered/$variant/$f")"
  done
done

mkdir -p "$(dirname "$OUT")"
{
  echo "{"
  echo "  \"commit\": \"$SHA\","
  echo "  \"checked_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
  echo "  \"files\": {"
  printf '    %s' "${results[0]}"
  for r in "${results[@]:1}"; do printf ',\n    %s' "$r"; done
  echo
  echo "  }"
  echo "}"
} > "$OUT"
cat "$OUT"
