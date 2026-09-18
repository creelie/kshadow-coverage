#!/usr/bin/env bash
#
# upload_kshadow.sh
#
# Pushes kshadow-coverage/ to GitHub and tags a release, which is what
# Zenodo mints the DOI from.
#
# Usage:
#   ./upload_kshadow.sh [-u USER] [-r REPO] [-d DIR] [-t TAG] [-p]
#
#   -u USER   GitHub account       (default: creelie)
#   -r REPO   repository name      (default: kshadow-coverage)
#   -d DIR    folder to push       (default: ./kshadow-coverage)
#   -t TAG    release tag          (default: v1.0.0)
#   -p        make the repository private (needs gh)
#
# It rewrites the placeholder URL in CITATION.cff and README.md to the
# account you pass, creates the repository with gh if gh is installed and
# logged in, and otherwise expects you to have created it in the browser
# first. Re-running it is safe: an existing git tree is updated, not wiped.

set -euo pipefail

USER_NAME="creelie"
REPO="kshadow-coverage"
DIR=""
TAG="v1.0.0"
VISIBILITY="--public"

while getopts "u:r:d:t:ph" opt; do
  case "$opt" in
    u) USER_NAME="$OPTARG" ;;
    r) REPO="$OPTARG" ;;
    d) DIR="$OPTARG" ;;
    t) TAG="$OPTARG" ;;
    p) VISIBILITY="--private" ;;
    h) sed -n '3,22p' "$0"; exit 0 ;;
    *) exit 2 ;;
  esac
done

# The script ships inside the repository, so default to the folder holding it;
# fall back to ./REPO when it has been copied out and sits beside the folder.
if [ -z "$DIR" ]; then
  HERE="$(cd "$(dirname "$0")" && pwd)"
  if [ -f "$HERE/README.md" ] && [ -f "$HERE/CITATION.cff" ]; then
    DIR="$HERE"
  else
    DIR="./$REPO"
  fi
fi
URL="https://github.com/$USER_NAME/$REPO"

command -v git >/dev/null || { echo "git is not installed."; exit 1; }
[ -d "$DIR" ] || { echo "No such folder: $DIR"; exit 1; }
cd "$DIR"
[ -f README.md ] || { echo "$DIR does not look like the repository (no README.md)."; exit 1; }

echo "==> repository : $URL"
echo "==> folder     : $(pwd)"
echo "==> tag        : $TAG"
echo

# 1. Point the placeholder URLs at the account actually being used.
if [ "$USER_NAME" != "creelie" ]; then
  for f in CITATION.cff README.md .zenodo.json; do
    [ -f "$f" ] || continue
    if grep -q "github.com/creelie/kshadow-coverage" "$f"; then
      sed -i.bak "s|github.com/creelie/kshadow-coverage|github.com/$USER_NAME/$REPO|g" "$f"
      rm -f "$f.bak"
      echo "    rewrote URL in $f"
    fi
  done
fi

# 2. Clear out build litter that .gitignore excludes anyway.
find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
find . -type f -name '*.pyc' -delete 2>/dev/null || true

# 3. Refuse anything GitHub will reject on size.
BIG=$(find . -type f -size +90M -not -path './.git/*' -print)
if [ -n "$BIG" ]; then
  echo "These files are over 90 MB and need Git LFS:"
  echo "$BIG"
  exit 1
fi

# 4. Local repository.
if [ ! -d .git ]; then
  git init -q
  git branch -M main
fi
if [ -z "$(git config user.name || true)" ] || [ -z "$(git config user.email || true)" ]; then
  echo "Git has no identity set. Run these once, then run this script again:"
  echo "    git config --global user.name  \"Deep Bhattacharjee\""
  echo "    git config --global user.email \"you@example.com\""
  exit 1
fi

git add -A
if git diff --cached --quiet; then
  echo "    nothing new to commit"
else
  git commit -q -m "k-shadow coverage certificates, ${TAG#v}"
  echo "    committed"
fi

# 5. Remote. gh creates the repository; without gh, create it in the browser first.
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$URL.git"
else
  if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
    gh repo view "$USER_NAME/$REPO" >/dev/null 2>&1 || \
      gh repo create "$USER_NAME/$REPO" $VISIBILITY \
        --description "Topological certificates of redundant coverage for cortical electrode arrays"
  fi
  git remote add origin "$URL.git"
fi

# 6. Push, then tag.
git push -u origin main
if git rev-parse "$TAG" >/dev/null 2>&1; then
  echo "    tag $TAG already exists locally"
else
  git tag -a "$TAG" -m "Version accompanying the manuscript"
fi
git push origin "$TAG"

echo
echo "Done. $URL"
echo "Next: switch the repository on at zenodo.org/account/settings/github,"
echo "then re-push the tag (or draft a release) so Zenodo mints the DOI."
