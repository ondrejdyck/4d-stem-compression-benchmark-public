#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
stage_dir="$repo_root/arxiv-src"
bundle_path="$repo_root/arxiv-src.tar.gz"

rm -rf "$stage_dir" "$bundle_path"
mkdir -p "$stage_dir/generated/tables" "$stage_dir/generated/figures" "$stage_dir/sections"

cp "$repo_root/paper/main.tex" "$stage_dir/main.tex"

REPO_ROOT="$repo_root" STAGE_DIR="$stage_dir" python3 - <<'PY'
import os
from pathlib import Path

repo_root = Path(os.environ["REPO_ROOT"])
stage_dir = Path(os.environ["STAGE_DIR"])

main_tex = stage_dir / "main.tex"
text = main_tex.read_text()
needle = "\\documentclass{article}\n"
insert = "\\documentclass{article}\n\\def\\publicrelease{}\n"
if needle not in text:
    raise SystemExit("Could not locate documentclass line in main.tex")
main_tex.write_text(text.replace(needle, insert, 1))

for src, dst in [
    (repo_root / "paper" / "sections", stage_dir / "sections"),
    (repo_root / "paper" / "generated" / "tables", stage_dir / "generated" / "tables"),
    (repo_root / "paper" / "generated" / "figures", stage_dir / "generated" / "figures"),
]:
    if dst.exists():
        for child in dst.iterdir():
            if child.is_file():
                child.unlink()
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.iterdir():
        if path.is_file() and path.suffix.lower() in {".tex", ".png", ".pdf", ".jpg", ".jpeg"}:
            (dst / path.name).write_bytes(path.read_bytes())

(stage_dir / "references.bib").write_bytes((repo_root / "paper" / "references.bib").read_bytes())
PY

tar -czf "$bundle_path" -C "$stage_dir" .

echo "Created staging directory: $stage_dir"
echo "Created archive: $bundle_path"
