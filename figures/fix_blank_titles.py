"""
Scan figures for blank white regions in title area and insert 'HypEReg-TransMorph'
text using insert_text (baseline-point method, more reliable than insert_textbox).
"""
import fitz, os

FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))
NEW = "HypEReg-TransMorph"

# For each figure, specify where the blank title hole is and what to insert.
# Coordinates from the original figure scans (before any patching):
#   fig2_qualitative: 'HER-TransMorph' bbox [475.22, 1.99, 677.81, 36.99] size 22 bold
#   fig3_gridwarp:    'HER-TransMorph' in first panel title, size 16 italic
#   fig4_jacobian:    'HER-TransMorph Jacobian' in first panel title, size 16 italic

TARGETS = {
    "fig2_qualitative.pdf": [
        # bbox was [475.22, 1.99, 677.81, 36.99], bold, size 22, centered column header
        # Use insert_text at baseline: x = center, y = bbox.y1 - 4
        {"text": NEW, "x": 475.0, "y": 32.0, "size": 22, "bold": True, "center_in": (475.22, 677.81)},
    ],
    "fig3_gridwarp.pdf": [
        # 'HER-TransMorph' title was similar to other two (TransMorph, MIDIR) - italic size 16
        # First panel is left-most; TransMorph x-center ~520, MIDIR x-center ~800
        # First panel x-center should be ~240 (rough estimate for 3-panel figure)
        {"text": NEW, "x": None, "y": None, "size": 16, "bold": False, "italic": True, "scan_first": True},
    ],
    "fig4_jacobian.pdf": [
        {"text": f"{NEW} Jacobian", "x": None, "y": None, "size": 16, "bold": False, "italic": True, "scan_first": True},
    ],
}


def get_spans_with_transmorph(pg):
    """Return all spans containing 'TransMorph' or 'MIDIR' (to infer position)."""
    spans = []
    for b in pg.get_text("dict")["blocks"]:
        for line in b.get("lines", []):
            for sp in line.get("spans", []):
                t = sp["text"].strip()
                if t and ("TransMorph" in t or "MIDIR" in t or "Jacobian" in t):
                    spans.append(sp)
    return spans


def center_x_of_bbox(bbox):
    return (bbox[0] + bbox[2]) / 2


def insert_centered_title(pg, text, cx, y, size, italic=False):
    """Insert text centered at cx, baseline at y."""
    fname = "tiro" if italic else "hebo"  # italic substitute
    # Actually use a font that supports italic: 'ti' (Times Italic) or just use regular
    # DejaVuSans was used originally; fall back to Helvetica
    fname = "helv"
    # Estimate text width at given size (rough: 0.55 * size * n_chars for Helvetica)
    est_width = 0.55 * size * len(text)
    x0 = cx - est_width / 2
    res = pg.insert_text(
        fitz.Point(x0, y),
        text,
        fontname=fname,
        fontsize=size,
        color=(0, 0, 0),
    )
    return res


def patch_fig(path, targets):
    doc = fitz.open(path)
    pg = doc[0]

    for cfg in targets:
        if cfg.get("scan_first"):
            # Determine position from sibling spans
            sibling_spans = get_spans_with_transmorph(pg)
            if not sibling_spans:
                print(f"  No sibling spans found in {os.path.basename(path)}, skipping.")
                continue
            # Sort by x position; first panel should have smallest x
            sibling_spans.sort(key=lambda s: s["bbox"][0])
            # Get y and size from the siblings (they share same baseline)
            sample = sibling_spans[0]
            y_baseline = sample["bbox"][3] - 2  # bbox bottom minus small offset
            y_baseline = sample["bbox"][1] + sample["size"] * 0.8  # approx baseline
            size = cfg["size"]
            # Infer x-center for first panel: 
            # In a 3-panel figure, panels are roughly equal width.
            # First sibling (now second panel) gives x-center for panel 2.
            # We need panel 1's x-center.
            pg_width = pg.rect.width
            # If there are 2 sibling panels found, estimate panel 1
            all_cx = sorted([center_x_of_bbox(s["bbox"]) for s in sibling_spans])
            if len(all_cx) >= 2:
                # gap between panel centers
                gap = all_cx[1] - all_cx[0]
                cx_p1 = all_cx[0] - gap
            else:
                cx_p1 = all_cx[0] - pg_width / 3
            cfg["x"] = cx_p1
            cfg["y"] = y_baseline

        cx = cfg["x"]
        if cfg.get("center_in"):
            # Center in given x range
            x_lo, x_hi = cfg["center_in"]
            cx = (x_lo + x_hi) / 2
        y = cfg["y"]
        size = cfg["size"]
        text = cfg["text"]
        italic = cfg.get("italic", False)

        res = insert_centered_title(pg, text, cx, y, size, italic)
        print(f"  Inserted {text!r} cx={cx:.1f} y={y:.1f} -> {res}")

    tmp = path + ".fix.tmp"
    doc.save(tmp)
    doc.close()
    os.replace(tmp, path)
    print(f"Saved: {os.path.basename(path)}")


if __name__ == "__main__":
    for fname, targets in TARGETS.items():
        path = os.path.join(FIGURES_DIR, fname)
        if not os.path.exists(path):
            print(f"Skip: {fname}")
            continue
        print(f"\n--- {fname} ---")
        patch_fig(path, targets)

    # Re-render verification strips
    import sys
    sys.path.insert(0, FIGURES_DIR)
    for fname in TARGETS:
        path = os.path.join(FIGURES_DIR, fname)
        doc = fitz.open(path)
        pg = doc[0]
        pr = pg.rect
        clip = fitz.Rect(0, 0, pr.width, 65)
        mat = fitz.Matrix(2, 2)
        pix = pg.get_pixmap(matrix=mat, clip=clip)
        out = os.path.join(FIGURES_DIR, fname.replace(".pdf", "_fixed_check.png"))
        pix.save(out)
        doc.close()
        print(f"Verification strip: {out}")
