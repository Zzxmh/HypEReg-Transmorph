"""
Check whether text was visually inserted (render to PNG), then try insert_text
at an absolute baseline position if insert_textbox isn't being found.
"""
import fitz
import os

FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))
OLD = "HER-TransMorph"
NEW = "HypEReg-TransMorph"

# Known positions from initial scan
# fig2: bbox [475.22, 1.99, 677.81, 36.99]  font DejaVuSans-Bold size 22  col_center x~576
# fig3: need to check
# fig4: two labels, need to check

def get_all_spans(path):
    doc = fitz.open(path)
    spans = []
    for pi, pg in enumerate(doc):
        for b in pg.get_text("dict")["blocks"]:
            for line in b.get("lines", []):
                for sp in line.get("spans", []):
                    spans.append({
                        "page": pi,
                        "text": sp["text"],
                        "bbox": sp["bbox"],
                        "font": sp["font"],
                        "size": sp["size"],
                        "color": sp["color"],
                    })
    doc.close()
    return spans


def render_strip(path, output_png, clip_rect):
    doc = fitz.open(path)
    pg = doc[0]
    mat = fitz.Matrix(2, 2)
    pix = pg.get_pixmap(matrix=mat, clip=fitz.Rect(clip_rect))
    pix.save(output_png)
    doc.close()
    print(f"  Rendered strip -> {output_png}")


def insert_text_at(path, text, point, size, bold=True, color=(0,0,0)):
    """Insert text at an absolute point (x, baseline_y) using insert_text."""
    doc = fitz.open(path)
    pg = doc[0]
    fname = "hebo" if bold else "helv"
    pg.insert_text(point, text, fontname=fname, fontsize=size, color=color)
    tmp = path + ".ins.tmp"
    doc.save(tmp)
    doc.close()
    os.replace(tmp, path)
    print(f"  Inserted {text!r} at {point} -> {os.path.basename(path)}")


if __name__ == "__main__":
    figs = {
        "fig2_qualitative.pdf": {
            # Column header, bbox was [475.22, 1.99, 677.81, 36.99]
            "point": (475, 30),  # x0, baseline_y (y1 - small offset)
            "size": 22,
            "bold": True,
        },
        "fig3_gridwarp.pdf": {
            "point": None,  # will scan
            "size": 16,
            "bold": False,
        },
        "fig4_jacobian.pdf": {
            "point": None,  # will scan
            "size": 16,
            "bold": False,
        },
    }

    for fname, cfg in figs.items():
        path = os.path.join(FIGURES_DIR, fname)
        if not os.path.exists(path):
            print(f"Skip (not found): {fname}")
            continue

        # First render strip to visually inspect
        strip_out = os.path.join(FIGURES_DIR, f"{fname}_strip.png")
        render_strip(path, strip_out, (0, 0, 900, 50))

        # Check current text
        spans = get_all_spans(path)
        title_spans = [s for s in spans if "TransMorph" in s["text"] or "HypEReg" in s["text"]]
        print(f"\n{fname}: title spans = {[s['text'] for s in title_spans]}")

        if cfg["point"] is not None:
            insert_text_at(path, NEW, cfg["point"], cfg["size"], cfg["bold"])
        
        # Re-verify
        spans2 = get_all_spans(path)
        print(f"  After insert: {[s['text'] for s in spans2 if 'TransMorph' in s['text'] or 'HypEReg' in s['text']]}")
