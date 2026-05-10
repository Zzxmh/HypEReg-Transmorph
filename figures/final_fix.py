"""
Final fix: correct title positions and font weights for all patched figures.
"""
import fitz, os

FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))


def remove_span_text(pg, text_fragment):
    """Redact and remove all spans containing text_fragment."""
    to_redact = []
    for b in pg.get_text("dict")["blocks"]:
        for line in b.get("lines", []):
            for sp in line.get("spans", []):
                if text_fragment in sp["text"]:
                    bbox = fitz.Rect(sp["bbox"])
                    # Expand slightly to fully cover
                    to_redact.append(fitz.Rect(bbox.x0 - 2, bbox.y0 - 2, bbox.x1 + 2, bbox.y1 + 2))
    for r in to_redact:
        pg.add_redact_annot(r, fill=(1, 1, 1))
    if to_redact:
        pg.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
    return len(to_redact)


def insert_oblique(pg, text, cx, baseline_y, size, color=(0, 0, 0)):
    """Insert italic-style text centered at cx."""
    # DejaVuSans-Oblique not available in standard fitz fonts;
    # use 'ti' (Times Italic) which renders as italic
    fname = "tiit"  # Times-Italic (base14 font)
    est_width = 0.52 * size * len(text)
    x0 = cx - est_width / 2
    pg.insert_text(fitz.Point(x0, baseline_y), text, fontname=fname, fontsize=size, color=color)


def insert_bold(pg, text, cx, baseline_y, size, color=(0, 0, 0)):
    """Insert bold text centered at cx."""
    fname = "hebo"  # Helvetica-Bold
    est_width = 0.60 * size * len(text)
    x0 = cx - est_width / 2
    pg.insert_text(fitz.Point(x0, baseline_y), text, fontname=fname, fontsize=size, color=color)


# ─────────────────────────────────────────────
# fig2_qualitative.pdf
# ─────────────────────────────────────────────
def fix_fig2():
    path = os.path.join(FIGURES_DIR, "fig2_qualitative.pdf")
    doc = fitz.open(path)
    pg = doc[0]
    # Remove any previously inserted (mispositioned or wrong-font) HypEReg text
    remove_span_text(pg, "HypEReg-TransMorph")
    # Original 'HER-TransMorph' was bbox [475.22, 1.99, 677.81, 36.99] DejaVuSans-Bold size 22
    # Center x = (475.22 + 677.81)/2 = 576.5, baseline_y ≈ y1 - 3 = 34
    insert_bold(pg, "HypEReg-TransMorph", cx=576.5, baseline_y=30.0, size=22)
    tmp = path + ".tmp"
    doc.save(tmp); doc.close(); os.replace(tmp, path)
    print("Fixed: fig2_qualitative.pdf")


# ─────────────────────────────────────────────
# fig3_gridwarp.pdf
# ─────────────────────────────────────────────
def fix_fig3():
    path = os.path.join(FIGURES_DIR, "fig3_gridwarp.pdf")
    doc = fitz.open(path)
    pg = doc[0]
    remove_span_text(pg, "HypEReg-TransMorph")
    # Sibling span 'TransMorph' should be at same y; scan for it
    sibs = []
    for b in pg.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            for s in l.get("spans", []):
                if "TransMorph" in s["text"] or "MIDIR" in s["text"]:
                    sibs.append(s)
    if sibs:
        # Sort by x to get leftmost (panel 2 and 3)
        sibs.sort(key=lambda s: s["bbox"][0])
        sample = sibs[0]
        baseline_y = sample["bbox"][1] + sample["size"] * 0.80
        size = sample["size"]  # 16
        # Infer panel 1 cx
        cx_sib0 = (sibs[0]["bbox"][0] + sibs[0]["bbox"][2]) / 2
        cx_sib1 = (sibs[1]["bbox"][0] + sibs[1]["bbox"][2]) / 2
        gap = cx_sib1 - cx_sib0
        cx_p1 = cx_sib0 - gap
    else:
        baseline_y, size, cx_p1 = 22, 16, 174
    insert_oblique(pg, "HypEReg-TransMorph", cx=cx_p1, baseline_y=baseline_y, size=size)
    tmp = path + ".tmp"
    doc.save(tmp); doc.close(); os.replace(tmp, path)
    print(f"Fixed: fig3_gridwarp.pdf  (cx={cx_p1:.1f} y={baseline_y:.1f})")


# ─────────────────────────────────────────────
# fig4_jacobian.pdf
# ─────────────────────────────────────────────
def fix_fig4():
    path = os.path.join(FIGURES_DIR, "fig4_jacobian.pdf")
    doc = fitz.open(path)
    pg = doc[0]
    # Remove misplaced "HypEReg-TransMorph Jacobian" at wrong y
    remove_span_text(pg, "HypEReg-TransMorph")
    # Sibling: 'TransMorph Jacobian' at bbox [322.7, 6.2, 489.3, 28.9] size 16
    #          'MIDIR Jacobian'      at bbox [590.4, 6.2, 709.9, 28.9] size 16
    # Panel 1 cx: spacing between panels is similar, page width 864
    # cx_panel2 = (322.7+489.3)/2 = 406.0
    # cx_panel3 = (590.4+709.9)/2 = 650.15
    # gap = 650.15 - 406.0 = 244.15
    # cx_panel1 ≈ 406.0 - 244.15 = 161.85
    baseline_y = 6.2 + 16 * 0.80  # ≈ 19.0
    insert_oblique(pg, "HypEReg-TransMorph Jacobian", cx=161.85, baseline_y=19.0, size=16)
    tmp = path + ".tmp"
    doc.save(tmp); doc.close(); os.replace(tmp, path)
    print("Fixed: fig4_jacobian.pdf")


# ─────────────────────────────────────────────
# Render verification strips
# ─────────────────────────────────────────────
def render_strip(fname, clip_bottom=65):
    path = os.path.join(FIGURES_DIR, fname)
    doc = fitz.open(path)
    pg = doc[0]
    clip = fitz.Rect(0, 0, pg.rect.width, clip_bottom)
    pix = pg.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip)
    out = os.path.join(FIGURES_DIR, fname.replace(".pdf", "_v2_check.png"))
    pix.save(out)
    doc.close()
    print(f"  Strip: {os.path.basename(out)}")


if __name__ == "__main__":
    fix_fig2()
    fix_fig3()
    fix_fig4()
    for f in ["fig2_qualitative.pdf", "fig3_gridwarp.pdf", "fig4_jacobian.pdf"]:
        render_strip(f)
