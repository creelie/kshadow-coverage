#!/usr/bin/env bash
# build.sh -- figures and manuscript of ecog_coverage.tex.
#
#   ./build.sh           every TikZ figure to figures/*.png (600 dpi), then the PDF
#   ./build.sh figs      the figures only
#   ./build.sh fig3_fold the one figure named
#
# Needs pdflatex with TikZ/pgfplots, pdftoppm (poppler) and latexmk.
# The plot tables in tikz/data/ come from ../../make_ecog_paper_data.py, and
# the fold geometry from tikz/fold_geometry.py; both are committed, so the
# figures build without Python.
set -euo pipefail
cd "$(dirname "$0")"
DPI=${DPI:-600}
mkdir -p tikz/build figures

build_fig() {
  local name=$1
  ( cd tikz && pdflatex -interaction=nonstopmode -halt-on-error \
      -output-directory=build "$name.tex" > "build/$name.stdout" 2>&1 ) || {
    echo "FAILED: $name (see tikz/build/$name.log)"; tail -30 "tikz/build/$name.log"; exit 1; }
  pdftoppm -png -r "$DPI" -singlefile "tikz/build/$name.pdf" "figures/$name"
  echo "figures/$name.png"
}

if [[ $# -ge 1 && $1 != figs ]]; then
  build_fig "$1"; exit 0
fi
for f in tikz/fig*.tex; do
  build_fig "$(basename "$f" .tex)"
done
[[ ${1:-} == figs ]] && exit 0
latexmk -pdf -interaction=nonstopmode -halt-on-error ecog_coverage.tex > /dev/null
echo "ecog_coverage.pdf"
