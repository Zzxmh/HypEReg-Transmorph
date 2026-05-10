"""
Build fig6 (downstream segmentation): 3 rows x 4 columns (GT, HypEReg, TransMorph, MIDIR).

Requires OASIS `data/Test_nii` volumes, atlas pickles, checkpoints, and a GPU
for reasonable runtime. Cached fused maps: `OASIS/Eval_Results/downstream/fused_cache/`.

Table-only bar chart (no data): `python figures/visualize_downstream_segmentation_bars.py`
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    test_nii = REPO_ROOT / "OASIS" / "data" / "Test_nii"
    if not test_nii.is_dir() or not any(test_nii.glob("img*.nii.gz")):
        print(
            "Missing OASIS test images under OASIS/data/Test_nii/.\n"
            "Cannot build qualitative fig6 here. Options:\n"
            "  • On a machine with data: python figures/render_downstream_qualitative.py --device cuda\n"
            "  • Bar summary (no MRI):     python figures/visualize_downstream_segmentation_bars.py",
            file=sys.stderr,
        )
        sys.exit(1)

    import torch

    sys.path.insert(0, str(REPO_ROOT / "figures"))
    from render_downstream_qualitative import render_figure

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir = REPO_ROOT / "OASIS" / "Eval_Results" / "downstream" / "fused_cache"
    out_pdf, out_png = render_figure(
        target_ids=[440, 444, 448],
        device=device,
        slice_axis=2,
        slice_index=None,
        cache_dir=cache_dir,
        from_cache_only=False,
        force=False,
        out_dir=REPO_ROOT / "figures",
    )
    print(f"Wrote: {out_pdf}\nWrote: {out_png}")


if __name__ == "__main__":
    main()
