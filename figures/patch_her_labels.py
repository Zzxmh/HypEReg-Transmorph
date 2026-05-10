"""Replace 'HER-TransMorph' with 'HypEReg-TransMorph' in all manuscript figure PDFs."""
import fitz
import os

OLD = "HER-TransMorph"
NEW = "HypEReg-TransMorph"

FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))
FIGS = [
    os.path.join(FIGURES_DIR, "fig2_qualitative.pdf"),
    os.path.join(FIGURES_DIR, "fig3_gridwarp.pdf"),
    os.path.join(FIGURES_DIR, "fig4_jacobian.pdf"),
    os.path.join(FIGURES_DIR, "fig5_metrics.pdf"),
]


def patch_pdf(path: str) -> None:
    doc = fitz.open(path)
    any_change = False

    for pg in doc:
        spans_to_replace = []
        for b in pg.get_text("dict")["blocks"]:
            for line in b.get("lines", []):
                for sp in line.get("spans", []):
                    if OLD not in sp["text"]:
                        continue
                    spans_to_replace.append(
                        {
                            "bbox": fitz.Rect(sp["bbox"]),
                            "size": sp["size"],
                            "bold": "Bold" in sp.get("font", ""),
                            "color": sp["color"],
                            "new_text": sp["text"].replace(OLD, NEW),
                        }
                    )

        if not spans_to_replace:
            continue
        any_change = True

        # --- pass 1: add all redaction annotations (expanded bbox to make room) ---
        for sp in spans_to_replace:
            b = sp["bbox"]
            expanded = fitz.Rect(b.x0 - 20, b.y0 - 2, b.x1 + 65, b.y1 + 2)
            pg.add_redact_annot(expanded, fill=(1, 1, 1))
        pg.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

        # --- pass 2: insert replacement text ---
        for sp in spans_to_replace:
            b = sp["bbox"]
            expanded = fitz.Rect(b.x0 - 20, b.y0 - 2, b.x1 + 65, b.y1 + 2)
            c = sp["color"]
            r_ = ((c >> 16) & 0xFF) / 255.0
            g_ = ((c >> 8) & 0xFF) / 255.0
            bv_ = (c & 0xFF) / 255.0
            fname = "hebo" if sp["bold"] else "helv"
            rc = pg.insert_textbox(
                expanded,
                sp["new_text"],
                fontname=fname,
                fontsize=sp["size"],
                color=(r_, g_, bv_),
                align=1,  # centered
            )
            status = "OK" if rc >= 0 else f"TRUNCATED (rc={rc})"
            print(f"  [{status}] {sp['new_text']!r} @ size={sp['size']:.0f}")

    if any_change:
        tmp = path + ".tmp"
        doc.save(tmp)
        doc.close()
        os.replace(tmp, path)
        print(f"Saved: {os.path.basename(path)}")
    else:
        doc.close()
        print(f"No 'HER-TransMorph' found: {os.path.basename(path)}")


def verify(path: str) -> None:
    doc = fitz.open(path)
    remain = []
    new_labels = []
    for pg in doc:
        for b in pg.get_text("dict")["blocks"]:
            for line in b.get("lines", []):
                for sp in line.get("spans", []):
                    t = sp["text"].strip()
                    if OLD in t:
                        remain.append(t)
                    if "HypEReg" in t:
                        new_labels.append(t)
    doc.close()
    print(f"  Remaining '{OLD}': {remain}")
    print(f"  New 'HypEReg' labels: {new_labels}")


if __name__ == "__main__":
    for p in FIGS:
        if not os.path.exists(p):
            print(f"SKIP (not found): {os.path.basename(p)}")
            continue
        print(f"\n--- {os.path.basename(p)} ---")
        patch_pdf(p)
        verify(p)
