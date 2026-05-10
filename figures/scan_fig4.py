import fitz
p = r"F:\TransMorph_Transformer_for_Medical_Image_Registration\TransMorph_Transformer_for_Medical_Image_Registration\figures\fig4_jacobian.pdf"
doc = fitz.open(p)
pg = doc[0]
print("Page size:", pg.rect)
for b in pg.get_text("dict")["blocks"]:
    for l in b.get("lines", []):
        for s in l.get("spans", []):
            t = s["text"].strip()
            if t and len(t) > 2:
                bbox = [round(x, 1) for x in s["bbox"]]
                print(f"  text={t!r:<40} bbox={bbox}  size={round(s['size'],1)} font={s['font']}")
doc.close()
