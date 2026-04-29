from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "draft"
SRC = DRAFT / "template.tex"
OUT_WITH_OASIS = DRAFT / "template_with_oasis.tex"
OUT_IXI_ONLY = DRAFT / "template_ixi_only.tex"


def _replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise ValueError(f"Expected snippet not found:\n{old[:120]}...")
    return text.replace(old, new, 1)


def make_ixi_only(tex: str) -> str:
    # Abstract: keep IXI-focused summary only.
    abstract_ixi = (
        r"\abstract{Background: Deformable brain MRI registration requires preserving deformation topology in addition to overlap accuracy. "
        r"Methods: HER-TransMorph augments a TransMorph backbone with hyperelastic regularization terms targeting volume-change moderation and anti-folding control. "
        r"We evaluate the method on the IXI atlas-to-subject protocol against learning-based and classical baselines, and report paired Wilcoxon signed-rank tests with Benjamini--Hochberg FDR correction where available. "
        r"Results: HER-TransMorph maintains competitive overlap (Dice: 0.7537 \(\pm\) 0.0275) while reducing topology violations (non-positive Jacobian ratio: 0.0000 vs.\ 0.0153 for TransMorph) and improving Jacobian regularity (SDlogJ: 0.3280 vs.\ 0.4920). "
        r"Conclusions: HER-TransMorph yields topology-preserving deformations on IXI, supporting downstream Jacobian-based morphometric analysis.}"
    )
    tex = re.sub(
        r"\\abstract\{Background:.*?analysis\.\}",
        lambda _m: abstract_ixi,
        tex,
        count=1,
        flags=re.S,
    )

    # Contribution bullet: remove OASIS statement.
    tex = _replace_once(
        tex,
        r"\item Subject-level paired statistical validation (Wilcoxon + BH-FDR), with dual-protocol evaluation across IXI and OASIS (\PH{OASIS stats pending Stage B}), to test robustness of topology-preserving behavior.",
        r"\item Subject-level paired statistical validation (Wilcoxon + BH-FDR) on the unified IXI protocol to test robustness of topology-preserving behavior.",
    )

    # Remove OASIS dataset paragraph in Methods.
    tex = re.sub(
        r"\nTo assess generalization beyond IXI atlas-to-subject, we additionally evaluate on OASIS.*?atlas-to-subject evaluation\.\n",
        "\n",
        tex,
        count=1,
        flags=re.S,
    )

    # Results text: remove OASIS table reference.
    tex = _replace_once(
        tex,
        r"Table~\ref{tab2} reports the compact IXI model comparison, now expanded to include overlap and regularity metrics in one place. HER-TransMorph maintains competitive overlap while improving deformation regularity relative to unconstrained Transformer baselines. Table~\ref{tab_oasis} mirrors the same panel for OASIS and is currently presented with explicit placeholders pending Stage-B OASIS training/inference completion.",
        r"Table~\ref{tab2} reports the compact IXI model comparison, expanded to include overlap and regularity metrics in one place. HER-TransMorph maintains competitive overlap while improving deformation regularity relative to unconstrained Transformer baselines.",
    )

    # Remove full OASIS table block.
    tex = re.sub(
        r"\n\\begin\{table\}\[H\]\n\\caption\{OASIS.*?\\end\{table\}\n",
        "\n",
        tex,
        count=1,
        flags=re.S,
    )

    # Figure captions / TODOs mentioning OASIS.
    tex = _replace_once(
        tex,
        r"\caption{Qualitative registration comparisons on representative IXI and OASIS subjects. \PH{figure regenerated in Stage B from saved deformation fields}.\label{fig2}}",
        r"\caption{Qualitative registration comparisons on representative IXI subjects. \PH{figure regenerated in Stage B from saved deformation fields}.\label{fig2}}",
    )
    tex = _replace_once(
        tex,
        r"% TODO Stage B: regenerate with IXI+OASIS dual panel metrics.",
        r"% TODO Stage B: regenerate with IXI panel metrics.",
    )
    tex = _replace_once(
        tex,
        r"\caption{Consolidated metric overview across selected methods (Dice, non\_jac, SDlogJ, bending energy, and runtime). \PH{figure regenerated in Stage B from IXI+OASIS panel data}.\label{fig5}}",
        r"\caption{Consolidated metric overview across selected methods (Dice, non\_jac, SDlogJ, bending energy, and runtime). \PH{figure regenerated in Stage B from IXI panel data}.\label{fig5}}",
    )

    # Discussion / limitations: remove OASIS-specific paragraphs and wording.
    tex = _replace_once(
        tex,
        r"The CNN/B-spline baseline MIDIR also achieves zero non-positive Jacobian ratio on IXI, with SDlogJ (0.3148) marginally lower than HER-TransMorph (0.3280). The two methods reach comparable regularity through different routes: MIDIR enforces smoothness implicitly via B-spline parameterization, while HER-TransMorph enforces regularity explicitly through differentiable hyperelastic penalties on dense flow. HER is plug-in to dense-flow backbones without changing parameterization, and preserves overlap expressivity (Dice 0.7537 vs.\ 0.7423 for MIDIR on IXI). Cross-dataset evidence on OASIS is summarized as \PH{OASIS MIDIR-vs-HER discussion pending Stage B}.",
        r"The CNN/B-spline baseline MIDIR also achieves zero non-positive Jacobian ratio on IXI, with SDlogJ (0.3148) marginally lower than HER-TransMorph (0.3280). The two methods reach comparable regularity through different routes: MIDIR enforces smoothness implicitly via B-spline parameterization, while HER-TransMorph enforces regularity explicitly through differentiable hyperelastic penalties on dense flow. HER is plug-in to dense-flow backbones without changing parameterization, and preserves overlap expressivity (Dice 0.7537 vs.\ 0.7423 for MIDIR on IXI).",
    )
    tex = re.sub(
        r"\nThe OASIS inter-subject regime produces larger anatomical displacements.*?hyperelastic-regularization mechanism\.\n",
        "\n",
        tex,
        count=1,
        flags=re.S,
    )
    tex = _replace_once(
        tex,
        r"Several limitations should be acknowledged. First, this manuscript retains a single training seed; multi-seed confidence intervals are future work. Second, a per-term ablation isolating \(\mathcal{L}_{\mathrm{length}}\), \(\mathcal{L}_{\mathrm{volume}}\), and \(\mathcal{L}_{\mathrm{fold}}\) was not rerun under the final operating-point pipeline; instead, cross-dataset OASIS validation is used as the main generalization check. Third, although IXI and OASIS cover two protocols, the study is still limited to structural brain MRI and does not yet include pathology-rich or multi-modal settings. Fourth, sensitivity sweeps for \(\beta\) and \(\gamma\) were not performed, and \(\alpha=0\) is a deliberate operating-point choice. Fifth, software-stack reproducibility against a stable PyTorch release is tracked as \PH{stable-PyTorch verification pending Stage B}.",
        r"Several limitations should be acknowledged. First, this manuscript retains a single training seed; multi-seed confidence intervals are future work. Second, a per-term ablation isolating \(\mathcal{L}_{\mathrm{length}}\), \(\mathcal{L}_{\mathrm{volume}}\), and \(\mathcal{L}_{\mathrm{fold}}\) was not rerun under the final operating-point pipeline. Third, the study is currently limited to structural brain MRI and does not yet include pathology-rich or multi-modal settings. Fourth, sensitivity sweeps for \(\beta\) and \(\gamma\) were not performed, and \(\alpha=0\) is a deliberate operating-point choice. Fifth, software-stack reproducibility against a stable PyTorch release is tracked as \PH{stable-PyTorch verification pending Stage B}.",
    )

    # Guardrail: no OASIS/Learn2Reg/table ref remains.
    forbidden = [r"\bOASIS\b", r"\boasis\b", r"Learn2Reg", r"tab_oasis"]
    for pat in forbidden:
        m = re.search(pat, tex)
        if m:
            ctx = tex[max(0, m.start() - 120) : min(len(tex), m.end() + 120)]
            raise ValueError(
                f"Found forbidden pattern in IXI-only draft: {pat}\nContext:\n{ctx}"
            )
    return tex


def main() -> None:
    source = SRC.read_text(encoding="utf-8")
    OUT_WITH_OASIS.write_text(source, encoding="utf-8")
    OUT_IXI_ONLY.write_text(make_ixi_only(source), encoding="utf-8")
    print(f"Wrote: {OUT_WITH_OASIS}")
    print(f"Wrote: {OUT_IXI_ONLY}")


if __name__ == "__main__":
    main()
