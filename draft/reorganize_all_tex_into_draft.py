"""
Move all LaTeX sources and common manuscript assets under draft/.

- Moves entire directories: Definitions/, submission_package/, submission_draft_backup_*/,
  _mdpi_template_extract/ (when present at repo root).
- Moves refs.bib and all root-level template* (sources + build artifacts) into draft/.
- Moves every other *.tex under the repo into draft/repository_tex/<relative/path>.tex
  (skips paths already inside draft/ and skips .git/.venv).

Run from repo root:
  python draft/reorganize_all_tex_into_draft.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

SKIP_DIR_NAMES = {".git", ".venv", "__pycache__", "node_modules"}


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _should_skip_dir(p: Path) -> bool:
    return any(part in SKIP_DIR_NAMES for part in p.parts)


def ensure_graphicspath(tex_path: Path) -> None:
    text = tex_path.read_text(encoding="utf-8")
    if "\\graphicspath" in text:
        return
    lines = text.splitlines(keepends=True)
    insert_at = 0
    for i, line in enumerate(lines):
        if "\\documentclass" in line:
            insert_at = i + 1
            break
    snippet = (
        "\n% Figures live at repository root; this manuscript is compiled from draft/.\n"
        "\\graphicspath{{../}}\n"
    )
    lines.insert(insert_at, snippet)
    tex_path.write_text("".join(lines), encoding="utf-8")


def move_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    shutil.move(str(src), str(dst))


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    draft = root / "draft"
    draft.mkdir(parents=True, exist_ok=True)

    # 1) Move whole manuscript support trees from repo root.
    move_tree(root / "Definitions", draft / "Definitions")
    move_tree(root / "submission_package", draft / "submission_package")
    for p in sorted(root.glob("submission_draft_backup*")):
        if p.is_dir():
            move_tree(p, draft / p.name)
    move_tree(root / "_mdpi_template_extract", draft / "_mdpi_template_extract")

    # 2) Bibliography + root LaTeX build products / main source.
    refs = root / "refs.bib"
    if refs.exists():
        shutil.move(str(refs), str(draft / "refs.bib"))

    for p in sorted(root.glob("template*")):
        if p.is_file():
            shutil.move(str(p), str(draft / p.name))

    # 3) Move every other .tex file into draft/repository_tex/<relpath>
    mirror = draft / "repository_tex"
    for tex in root.rglob("*.tex"):
        if _should_skip_dir(tex):
            continue
        if _is_under(tex, draft):
            continue
        rel = tex.relative_to(root)
        dest = mirror / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tex), str(dest))

    main_tex = draft / "template.tex"
    if not main_tex.exists():
        print("ERROR: draft/template.tex missing after move.", file=sys.stderr)
        return 1

    ensure_graphicspath(main_tex)
    print("Reorganization complete. Main manuscript: draft/template.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
