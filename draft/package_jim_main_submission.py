# -*- coding: utf-8 -*-
"""Build a self-contained main-manuscript LaTeX zip for journal resubmission (not SM).

Output: draft/JIM_main_manuscript_latex_submission.zip
Unpack, then from the folder: pdflatex article.tex; bibtex article; pdflatex article.tex; pdflatex article.tex
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

DRAFT = Path(__file__).resolve().parent
ROOT = DRAFT.parent
OUT = DRAFT / "JIM_main_manuscript_latex_submission"
SRC_TEX = DRAFT / "article.tex"
FIGS = ROOT / "figures"


def patch_article_tex(text: str) -> str:
    # Flat layout: all paths relative to submission root (no ../ for assets)
    text = text.replace("\\graphicspath{{../}}", "\\graphicspath{{./}}")
    old_oasis = (
        "\\IfFileExists{../figures/fig_oasis_jacobian.png}%\n"
        "  {\\includegraphics[width=0.97\\textwidth]{../figures/fig_oasis_jacobian.png}}%\n"
        "  {\\includegraphics[width=0.97\\textwidth]{figures/fig_oasis_jacobian.png}}"
    )
    new_oasis = "\\includegraphics[width=0.97\\textwidth]{figures/fig_oasis_jacobian.png}"
    if old_oasis not in text:
        raise RuntimeError("article.tex: expected OASIS figure block not found; aborting.")
    text = text.replace(old_oasis, new_oasis)
    old_fig1 = (
        "\\IfFileExists{HER_final.png}{\\includegraphics[width=0.96\\textwidth]{HER_final.png}}"
        "{\\IfFileExists{../HER_final.png}{\\includegraphics[width=0.96\\textwidth]{../HER_final.png}}"
        "{\\includegraphics[width=0.96\\textwidth]{figures/fig1_framework_figma.png}}}"
    )
    new_fig1 = (
        "\\IfFileExists{HER_final.png}{\\includegraphics[width=0.96\\textwidth]{HER_final.png}}"
        "{\\includegraphics[width=0.96\\textwidth]{figures/fig1_framework_figma.png}}"
    )
    if old_fig1 not in text:
        raise RuntimeError("article.tex: expected Figure 1 IfFileExists block not found; aborting.")
    text = text.replace(old_fig1, new_fig1)
    return text


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    # Class / style / bst / logos (MDPI)
    shutil.copytree(DRAFT / "Definitions", OUT / "Definitions")

    # Bibliography
    shutil.copy2(DRAFT / "refs.bib", OUT / "refs.bib")

    # Main text (patched for flat bundle)
    tex = SRC_TEX.read_text(encoding="utf-8")
    OUT.joinpath("article.tex").write_text(patch_article_tex(tex), encoding="utf-8", newline="\n")

    # Figure 1 + fallback
    fig_out = OUT / "figures"
    fig_out.mkdir(parents=True, exist_ok=True)
    for name in (
        "fig3_gridwarp.png",
        "fig4_jacobian.png",
        "fig_oasis_jacobian.png",
        "fig1_framework_figma.png",
        "fig6_downstream_segmentation.png",
    ):
        shutil.copy2(FIGS / name, fig_out / name)

    her = DRAFT / "HER_final.png"
    if not her.is_file():
        raise RuntimeError("draft/HER_final.png missing; required for Figure 1 in submission bundle.")
    shutil.copy2(her, OUT / "HER_final.png")

    readme = OUT / "README_COMPILE.txt"
    readme.write_text(
        "Main manuscript only (no supplementary).\n\n"
        "Compile (pdfLaTeX + BibTeX):\n"
        "  pdflatex article.tex\n"
        "  bibtex article\n"
        "  pdflatex article.tex\n"
        "  pdflatex article.tex\n",
        encoding="utf-8",
        newline="\n",
    )

    zip_path = DRAFT / "JIM_main_manuscript_latex_submission.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in OUT.rglob("*"):
            if p.is_file():
                arc = p.relative_to(OUT).as_posix()
                zf.write(p, arc)
    print(f"Wrote folder: {OUT}")
    print(f"Wrote zip:    {zip_path} ({zip_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
