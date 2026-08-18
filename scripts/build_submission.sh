#!/usr/bin/env bash
# Build a strict, leak-proof contest submission archive (.tar.gz, which the
# harness accepts: "unzip submission.zip OR tar -xzf submission.tar.gz").
#
# WHY working-tree (not `git archive`): some optimizer/*.py modules used in
# --contest-mode (e.g. negative_memory.py) are currently UNTRACKED. `git
# archive HEAD` would silently omit them, shipping a submission that runs but
# with features disabled — differing from what was validated on AWS. We build
# from the working tree so the archive matches the validated runtime exactly.
#
# Strictness (the alpha-0.0 / "fails on their env" class):
#   - EXCLUDE .env  (leaks VIVADO_EXEC=/home/<dev>/.local/bin/vivado + local
#     JAVA_HOME + our API key -> AWS-fatal / rule-violating; harness sets
#     OPENROUTER_API_KEY itself)
#   - EXCLUDE .venv (a local python venv shipped onto the AWS python is wrong
#     and bloats the archive)
#   - EXCLUDE .git, run dirs, benchmarks (harness provides), validation dirs,
#     DCP/EDIF artifacts, caches, the RapidWright submodule (pip rapidwright
#     is used at runtime; build-rapidwright skips cleanly without .git)
#
# After building, VERIFY: no secrets/venv leaked, key files present, and the
# archive imports cleanly in a fresh extract (catches a missing untracked
# module before the contest harness does).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_NAME="$(basename "$REPO_DIR")"
OUT="${1:-/tmp/fpl26_submission.tar.gz}"

# The archive's TOP-LEVEL DIRECTORY is part of the contract: the harness
# extracts and enters `fpl26_optimization_contest/`. It used to be taken from
# the CHECKOUT dirname, so building from a git worktree (jul29:
# `wt_final_round_dev/`) produced an archive rooted at the worktree's name —
# and every check below, which derived its expected prefix from that SAME
# dirname, agreed with it and printed "SUBMISSION READY". The verifier was
# validating the build against itself rather than against the contract, so the
# only thing that ever caught it was tests/test_build_submission.py, whose
# failure had been written off as "a worktree artifact".
#
# Force the stored prefix instead of deriving it. SUBMISSION_ROOT overrides it
# for a deliberate rename.
ARCHIVE_ROOT="${SUBMISSION_ROOT:-fpl26_optimization_contest}"

# ---- PROVENANCE GATE: refuse to package a tree a newer branch supersedes ----
#
# The archive-root bug above was invisible from the MAIN checkout, because that
# directory happens to be named `fpl26_optimization_contest`. The mirror-image
# hazard is worse and is live: on jul29 the main checkout sat on
# `final-dev-portfolio-scheduler`, a STRICT ANCESTOR 140 commits behind
# `final-round-dev`. Packaging it would have shipped a tree with none of the
# final round in it — including the jul29 fix that put the uniform ILS stack on
# the ship path at all — and every check in this script would have passed,
# because the dirname is right and the files are all present. Correct-looking
# archive, wrong decade of code.
#
# So check the one thing the rest of this script cannot see: is some other local
# branch a strict descendant of what we are about to package? That fires only
# when a newer line of development demonstrably exists, so it does not nag on a
# normal release, and it cannot be satisfied by the build validating itself.
#
# SUBMISSION_ALLOW_STALE=1 overrides, for a deliberate build of an older tree.
if git -C "$REPO_DIR" rev-parse --git-dir >/dev/null 2>&1; then
  HEAD_SHA="$(git -C "$REPO_DIR" rev-parse --short HEAD)"
  HEAD_BRANCH="$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD)"
  DIRTY=""
  git -C "$REPO_DIR" diff --quiet 2>/dev/null || DIRTY=" (tracked files MODIFIED)"
  echo "[build_submission] packaging $HEAD_BRANCH @ $HEAD_SHA$DIRTY"

  NEWER=""
  while read -r b; do
    [ -n "$b" ] || continue
    # Skip any branch whose tip IS this commit — those are not "newer".
    [ "$(git -C "$REPO_DIR" rev-parse "$b" 2>/dev/null)" = \
      "$(git -C "$REPO_DIR" rev-parse HEAD)" ] && continue
    ahead="$(git -C "$REPO_DIR" rev-list --count "HEAD..$b" 2>/dev/null || echo '?')"
    NEWER="$NEWER
  $b is $ahead commits ahead of what you are packaging"
  done <<EOF
$(git -C "$REPO_DIR" branch --format='%(refname:short)' --contains HEAD 2>/dev/null \
    | grep -v "^$HEAD_BRANCH\$" || true)
EOF

  if [ -n "$NEWER" ] && [ "${SUBMISSION_ALLOW_STALE:-0}" != "1" ]; then
    echo "[build_submission] FAIL: this tree is superseded by a newer branch:$NEWER" >&2
    echo "" >&2
    echo "  Packaging a strict ancestor ships code that looks complete and is not." >&2
    echo "  Build from the newer branch, or set SUBMISSION_ALLOW_STALE=1 if this" >&2
    echo "  older tree is genuinely what you mean to submit." >&2
    exit 1
  fi
  [ -n "$NEWER" ] && echo "[build_submission] WARNING: superseded tree packaged on purpose (SUBMISSION_ALLOW_STALE=1):$NEWER"
fi

cd "$REPO_DIR/.."

# The --exclude patterns match ON-DISK names, so they keep using $REPO_NAME;
# --transform rewrites only the names STORED in the archive.
XFORM=()
if [ "$REPO_NAME" != "$ARCHIVE_ROOT" ]; then
  echo "[build_submission] checkout dir is '$REPO_NAME'; forcing archive root to '$ARCHIVE_ROOT'"
  # Anchored at ^, and S so symlink targets are left alone. Every member name
  # tar sees begins with "$REPO_NAME" because that is the single path handed
  # to it, so this cannot reach a sibling directory.
  XFORM=(--transform "s,^${REPO_NAME},${ARCHIVE_ROOT},S")
fi

echo "[build_submission] packaging $REPO_NAME -> $OUT (archive root: $ARCHIVE_ROOT)"
tar czf "$OUT" ${XFORM[@]+"${XFORM[@]}"} \
  --exclude="$REPO_NAME/.git" \
  --exclude="$REPO_NAME/.venv" \
  --exclude="$REPO_NAME/.env" \
  --exclude="$REPO_NAME/.env.*" \
  --exclude="$REPO_NAME/dcp_optimizer_run-*" \
  --exclude="$REPO_NAME/fpl26_contest_benchmarks" \
  --exclude="$REPO_NAME/submission" \
  --exclude="$REPO_NAME/dcp_validation_*" \
  --exclude='*.dcp' --exclude='*.edf' --exclude='__pycache__' --exclude='*.pyc' \
  --exclude="$REPO_NAME/.pytest_cache" \
  --exclude="$REPO_NAME/rqs_exp" --exclude="$REPO_NAME/results" \
  --exclude="$REPO_NAME/staging_v1.1.0" --exclude="$REPO_NAME/submissions" \
  --exclude="$REPO_NAME/RapidWright" \
  --exclude="$REPO_NAME/mlcad_benchmarks" --exclude="$REPO_NAME/baselines" \
  --exclude="$REPO_NAME/live_tests" \
  "$REPO_NAME"

echo "[build_submission] built: $(du -h "$OUT" | cut -f1)"

# ---- STRICT VERIFICATION -------------------------------------------------
fail() { echo "[build_submission] FAIL: $1" >&2; exit 1; }

# Capture the listing ONCE (piping into `grep -q` would SIGPIPE `tar` and, under
# `set -o pipefail`, register a false failure).
listing="$(tar tzf "$OUT")"

leaks="$(printf '%s\n' "$listing" | grep -iE "/\.env$|/\.env\.|\.venv/|\.pem$|fpl26contest-key|/\.git/" || true)"
[ -n "$leaks" ] && fail "secret/venv/git leak in archive:"$'\n'"$leaks"
echo "[build_submission] OK: no .env/.venv/.git/secret leaks"

# The archive root is a CONTRACT with the harness, so check it against the
# literal expected name — never against a value derived from this checkout.
strays="$(printf '%s\n' "$listing" | grep -vE "^${ARCHIVE_ROOT}(/|$)" || true)"
[ -n "$strays" ] && fail "archive members outside '$ARCHIVE_ROOT/':"$'\n'"$(printf '%s\n' "$strays" | head -20)"
echo "[build_submission] OK: every member is under $ARCHIVE_ROOT/"

for f in Makefile dcp_optimizer.py requirements.txt SYSTEM_PROMPT.TXT \
         optimizer/recipe_router.py VivadoMCP/vivado_mcp_server.py \
         scripts/multi_restart_optimize.py; do
  printf '%s\n' "$listing" | grep -qx "$ARCHIVE_ROOT/$f" || fail "missing required file: $f"
done
echo "[build_submission] OK: required files present"

# Import check: extract to a temp dir and import dcp_optimizer to catch a
# missing untracked module (ImportError) the way the harness would.
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
tar xzf "$OUT" -C "$TMP"
if python3 -c "import sys; sys.path.insert(0, '$TMP/$ARCHIVE_ROOT'); import dcp_optimizer" 2>/tmp/import_check.err; then
  echo "[build_submission] OK: dcp_optimizer imports cleanly from the archive"
else
  echo "[build_submission] import error:" >&2; cat /tmp/import_check.err >&2
  fail "archive does not import cleanly (a needed module is missing from the archive)"
fi

echo "[build_submission] SUBMISSION READY: $OUT"
