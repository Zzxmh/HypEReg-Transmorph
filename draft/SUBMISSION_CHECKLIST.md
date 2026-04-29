# Submission checklist (MDPI *Journal of Imaging* revision)

Status legend: done / partial / author TODO

## Editor-mandated manuscript fixes (mapped to repository)

| Item | Status | Notes |
| --- | --- | --- |
| Rename HER $\rightarrow$ HypEReg-TransMorph; remove HER2 footnote; abbreviations | done | `draft/template_ixi_only.tex`, `draft/template_ixi_only_arxiv.tex`, `draft/supplementary.tex` |
| Remove conflicting main-text Figure 5; point to Supplementary Figure S1 | done | Main + arXiv appendices; figure only in `draft/supplementary.tex` |
| Limitations: single-seed; defer multi-seed CIs | done | Discussion in both manuscripts |
| PyTorch nightly note + stable-release reproducibility | done | Training details (both manuscripts) |
| SyN efficiency footnote (not comparable to feed-forward) | done | Table 3 footnote (both manuscripts) |
| Data availability: GitHub + Zenodo upon acceptance | done | `\dataavailability` / arXiv Data Availability paragraph |
| Abstract numerics + Wilcoxon/BH-FDR sentence | done | Both abstracts |
| MDPI metadata (authors, TODO ORCID/email/phone, empty dates) | partial | Names filled; `% TODO` remains for ORCID, affiliations, correspondence |
| MDPI numeric citations `\cite{...}` | done | `draft/template_ixi_only.tex` |
| `refs.bib` (IXI access note, arXiv notes, DOIs, software misc) | done | `draft/refs.bib` |
| Algorithm 1 in `algorithm` / `algpseudocode` | done | MDPI manuscript only |
| Supplementary refresh (intro + S1/S2/S3 naming) | done | `draft/supplementary.tex` |

## Materials outside the main PDF

| Item | Status | Artifact / action |
| --- | --- | --- |
| Main journal PDF | done (after compile) | `draft/template_ixi_only.pdf` |
| arXiv PDF | done (after compile) | `draft/template_ixi_only_arxiv.pdf` |
| Supplementary PDF | done (after compile) | `draft/supplementary.pdf` |
| LaTeX source ZIP for journal | author TODO | Zip `draft/`, `Definitions/`, `refs.bib`, figures as required by MDPI |
| Cover letter | done (after compile) | `draft/cover_letter.tex` / `draft/cover_letter.pdf` |
| Recommended reviewers (SuSy) | author TODO | `draft/recommended_reviewers.md` placeholders |
| ORCID, email, phone in TeX | author TODO | `% TODO` in `\address`, `\corres`, ORCID macro |
| MDPI conflicts-of-interest form | author TODO | Download from MDPI, sign, upload |
| Graphical abstract | author TODO | Not included in repository |
| Zenodo DOI | deferred | Statement: assigned upon acceptance |
