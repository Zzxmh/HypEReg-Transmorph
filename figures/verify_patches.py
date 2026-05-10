"""Render small PNG strips from the top of each patched figure to verify visually."""
import fitz, os

FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))

figs = ["fig2_qualitative.pdf", "fig3_gridwarp.pdf", "fig4_jacobian.pdf", "fig5_metrics.pdf"]

for fname in figs:
    path = os.path.join(FIGURES_DIR, fname)
    if not os.path.exists(path):
        continue
    doc = fitz.open(path)
    pg = doc[0]
    pr = pg.rect
    # Render top ~55pt of the page
    clip = fitz.Rect(0, 0, pr.width, 60)
    mat = fitz.Matrix(2, 2)
    pix = pg.get_pixmap(matrix=mat, clip=clip)
    out = os.path.join(FIGURES_DIR, fname.replace(".pdf", "_title_check.png"))
    pix.save(out)
    doc.close()
    print(f"Rendered: {os.path.basename(out)}")
