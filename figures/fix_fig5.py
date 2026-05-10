"""Fix fig5_metrics.pdf: replace truncated 'HypEReg-Trans' tick labels with full name."""
import fitz, os

FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(FIGURES_DIR, "fig5_metrics.pdf")

doc = fitz.open(path)
pg = doc[0]

# Find all 'HypEReg-Trans' spans (truncated) and their bboxes
truncated_spans = []
for b in pg.get_text("dict")["blocks"]:
    for line in b.get("lines", []):
        for sp in line.get("spans", []):
            if "HypEReg-Trans" in sp["text"]:
                truncated_spans.append({
                    "bbox": fitz.Rect(sp["bbox"]),
                    "size": sp["size"],
                    "font": sp["font"],
                    "color": sp["color"],
                    "text": sp["text"],
                })

print(f"Found {len(truncated_spans)} truncated spans")
for s in truncated_spans:
    print(f"  {s['text']!r} @ {[round(x,1) for x in s['bbox']]} size={s['size']:.1f} font={s['font']}")

# Redact expanded areas
for sp in truncated_spans:
    b = sp["bbox"]
    # x-axis labels are rotated 30 degrees; bbox might be narrow
    # Expand significantly in all directions
    expanded = fitz.Rect(b.x0 - 5, b.y0 - 3, b.x1 + 55, b.y1 + 3)
    pg.add_redact_annot(expanded, fill=(1, 1, 1))
pg.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

# Re-insert with wider textbox
for sp in truncated_spans:
    b = sp["bbox"]
    c = sp["color"]
    r_ = ((c >> 16) & 0xFF) / 255.0
    g_ = ((c >>  8) & 0xFF) / 255.0
    bv_ = (c & 0xFF) / 255.0
    fname = "hebo" if "Bold" in sp["font"] else "helv"
    expanded = fitz.Rect(b.x0 - 5, b.y0 - 3, b.x1 + 55, b.y1 + 3)
    rc = pg.insert_textbox(expanded, "HypEReg-TransMorph",
                           fontname=fname, fontsize=sp["size"],
                           color=(r_, g_, bv_), align=0)
    print(f"  Inserted 'HypEReg-TransMorph' rc={rc}")

tmp = path + ".tmp"
doc.save(tmp)
doc.close()
os.replace(tmp, path)
print("Saved: fig5_metrics.pdf")

# Render full figure for verification
doc2 = fitz.open(path)
pg2 = doc2[0]
pr = pg2.rect
clip = fitz.Rect(0, pr.height * 0.7, pr.width, pr.height)
pix = pg2.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip)
out = os.path.join(FIGURES_DIR, "fig5_metrics_bottom_check.png")
pix.save(out)
doc2.close()
print(f"Rendered bottom strip: {out}")
