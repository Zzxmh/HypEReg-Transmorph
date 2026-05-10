# -*- coding: utf-8 -*-
"""Rasterize figure PDFs to full-page PNGs (same basename) for LaTeX builds.

Uses PyMuPDF. Default DPI=300 for manuscript-quality bitmaps.
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError as e:  # pragma: no cover
    print("PyMuPDF (fitz) is required: pip install pymupdf", file=sys.stderr)
    raise SystemExit(1) from e


def _effective_dpi(page, dpi: float, max_px: int) -> float:
    """Cap raster resolution so the longer bitmap edge does not exceed max_px."""
    r = page.rect
    w_pt, h_pt = float(r.width), float(r.height)
    if w_pt <= 0 or h_pt <= 0:
        return dpi
    need = max(w_pt, h_pt) * (dpi / 72.0)
    if need <= max_px:
        return dpi
    return dpi * (max_px / need)


def pdf_to_png(pdf_path: Path, png_path: Path, dpi: float = 300.0, max_px: int = 5000) -> None:
    doc = fitz.open(pdf_path)
    if doc.page_count < 1:
        print(f"skip (no pages): {pdf_path.name}")
        return
    # Single-page figures in this repo; multi-page -> one PNG per page
    if doc.page_count == 1:
        page = doc[0]
        dpi_use = _effective_dpi(page, dpi, max_px)
        zoom = dpi_use / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        pix.save(str(png_path))
        print(f"wrote {png_path} ({pix.width}x{pix.height}) @~{dpi_use:.0f}dpi")
    else:
        stem = png_path.stem
        parent = png_path.parent
        for i in range(doc.page_count):
            page = doc[i]
            dpi_use = _effective_dpi(page, dpi, max_px)
            zoom = dpi_use / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            out = parent / f"{stem}_p{i + 1:02d}.png"
            pix.save(str(out))
            print(f"wrote {out} ({pix.width}x{pix.height}) page {i + 1}/{doc.page_count}")
    doc.close()


def main() -> None:
    root = Path(__file__).resolve().parent
    dpi = 300.0
    # All manuscript-style figure PDFs in this folder
    pdfs = sorted(
        p
        for p in root.glob("fig*.pdf")
        if p.is_file() and not p.name.startswith(".")
    )
    if not pdfs:
        print("no fig*.pdf found", file=sys.stderr)
        raise SystemExit(1)
    for pdf in pdfs:
        png = pdf.with_suffix(".png")
        pdf_to_png(pdf, png, dpi=dpi)


if __name__ == "__main__":
    main()
