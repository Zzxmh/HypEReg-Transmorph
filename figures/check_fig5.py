import fitz, os
FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(FIGURES_DIR, "fig5_metrics.pdf")
doc = fitz.open(path)
pg = doc[0]
pr = pg.rect
clip = fitz.Rect(0, pr.height * 0.65, pr.width, pr.height)
pix = pg.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip)
pix.save(os.path.join(FIGURES_DIR, "fig5_regen_check.png"))
doc.close()
print("Done")
