# -*- coding: utf-8 -*-
"""Self-contained supplementary Materials LaTeX bundle for journal upload.

Output: draft/JIM_supplementary_materials_latex_submission/
Unpack from that folder: pdflatex supplementary.tex (twice recommended).

Unlike the main manuscript, SM uses standard article.cls only (no MDPI Definitions).
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

DRAFT = Path(__file__).resolve().parent
ROOT = DRAFT.parent
FIGS = ROOT / "figures"
OUT = DRAFT / "JIM_supplementary_materials_latex_submission"
SRC_TEX = DRAFT / "supplementary.tex"


def patch_supplementary_tex(text: str) -> str:
    text = text.replace(
        "\\graphicspath{{../}{../figures/}}",
        "\\graphicspath{{./}{figures/}}",
    )
    return text


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    tex = SRC_TEX.read_text(encoding="utf-8")
    OUT.joinpath("supplementary.tex").write_text(patch_supplementary_tex(tex), encoding="utf-8", newline="\n")

    fig_out = OUT / "figures"
    fig_out.mkdir(parents=True, exist_ok=True)
    for name in ("fig5_metrics.png", "fig2_qualitative.png"):
        shutil.copy2(FIGS / name, fig_out / name)

    (OUT / "README_COMPILE.txt").write_text(
        "Supplementary Materials only.\n\n"
        "Compile with pdfLaTeX (no BibTeX):\n"
        "  pdflatex supplementary.tex\n"
        "  pdflatex supplementary.tex\n",
        encoding="utf-8",
        newline="\n",
    )

    zip_path = DRAFT / "JIM_supplementary_materials_latex_submission.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in OUT.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(OUT).as_posix())

    # Figures-only zip (both SM figures)
    fig_zip = DRAFT / "JIM_supplementary_materials_figures_only.zip"
    if fig_zip.exists():
        fig_zip.unlink()
    with zipfile.ZipFile(fig_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in ("fig5_metrics.png", "fig2_qualitative.png"):
            p = fig_out / name
            zf.write(p, f"figures/{name}")

    print(f"Wrote folder: {OUT}")
    print(f"Wrote zip:    {zip_path} ({zip_path.stat().st_size} bytes)")
    print(f"Wrote figs:   {fig_zip} ({fig_zip.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
