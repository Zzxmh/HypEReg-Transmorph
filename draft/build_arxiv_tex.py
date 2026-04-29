# -*- coding: utf-8 -*-
"""Generate draft/template_ixi_only_arxiv.tex from template_ixi_only.tex (single source for body)."""
from __future__ import annotations

import re
from pathlib import Path

DRAFT = Path(__file__).resolve().parent
SRC = DRAFT / "template_ixi_only.tex"
OUT = DRAFT / "template_ixi_only_arxiv.tex"

PREAMBLE = r"""% arXiv-oriented preprint (standard article class).
% Body text is derived from template_ixi_only.tex (same scientific content; MDPI layout removed).
\pdfoutput=1

\documentclass[11pt,a4paper]{article}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{float}
\usepackage[hyphens]{url}
\usepackage[margin=1in]{geometry}
\usepackage{xcolor}
\usepackage[round,authoryear]{natbib}
\usepackage[colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue]{hyperref}
\usepackage{enumitem}

\graphicspath{{../}{../figures/}}

\title{HypEReg-TransMorph: A Hyperelastic-Regularized Transformer\\ Framework for Folding-Suppressed Deformable Brain MRI Registration}

\author{%
Author One$^{1}$,
Author Two$^{2}$,
Corresponding Author$^{2,*}$\\[0.75em]
\small
$^{1}$ Department of Biomedical Engineering, Institution One, City, Country; author1@example.edu\\
$^{2}$ Department of Medical Imaging, Institution Two, City, Country; author2@example.edu\\[0.5em]
\small\textit{Correspondence:} corresponding.author@example.edu; Tel.: +00-000000000
}

\date{April 2026}

\begin{document}
\maketitle

\begin{abstract}
Background: Deformable brain MRI registration requires accurate alignment and geometrically plausible deformation fields. Methods: Hyperelastic Regularization (HER; retained acronym with explicit HER2 disambiguation) is integrated into a TransMorph-style dense registration backbone to penalize implausible local volume change and folding. Evaluation is performed on 115 held-out atlas-to-subject pairs from the Information eXtraction from Images (IXI) dataset using grouped anatomical labels. Results: HER-TransMorph yields statistically indistinguishable grouped-VOI Dice relative to the strongest Transformer baselines (0.7537 $\pm$ 0.0275 vs.\ 0.7530 $\pm$ 0.0302 for TransMorphBayes) while reducing the non-positive Jacobian-determinant ratio by approximately three orders of magnitude (0.000015 $\pm$ 0.000007 vs.\ 0.015634 $\pm$ 0.003363) and lowering SDlogJ (0.3280 $\pm$ 0.0221 vs.\ 0.4920 $\pm$ 0.0330). Conclusions: Under the IXI atlas-to-subject protocol, HER-TransMorph improves deformation regularity with near-equal overlap, supporting Jacobian-aware neuroimaging analysis.
\end{abstract}

\noindent\textbf{Keywords:} medical image registration; deformable registration; transformer; hyperelastic regularization; folding suppression; Jacobian regularization; Jacobian determinant; brain MRI

\vspace{0.5em}
\hrule
\vspace{0.8em}

"""


POSTAMBLE = r"""
\bibliographystyle{plainnat}
\bibliography{refs}

\end{document}
"""


def _extract_braced(src: str, start_open: int) -> tuple[str, int]:
    """Return (inner, index_after_closing_brace). start_open points at '{'."""
    if start_open < 0 or start_open >= len(src) or src[start_open] != "{":
        raise ValueError("expected '{'")
    depth = 0
    for i in range(start_open, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start_open + 1 : i], i + 1
    raise ValueError("unbalanced braces")


def replace_command(body: str, cmd: str, replacement_prefix: str) -> str:
    r"""Replace \cmd{...} with replacement_prefix + content (braces stripped)."""
    token = f"\\{cmd}{{"
    out: list[str] = []
    i = 0
    while i < len(body):
        j = body.find(token, i)
        if j == -1:
            out.append(body[i:])
            break
        out.append(body[i:j])
        open_idx = j + len(token) - 1
        inner, after = _extract_braced(body, open_idx)
        out.append(replacement_prefix + inner)
        i = after
    return "".join(out)


def replace_abbreviations_mdpi(body: str) -> str:
    """\\abbreviations{Title}{Body} -> \\section*{Title}\\n\\nBody"""
    key = "\\abbreviations{"
    out: list[str] = []
    i = 0
    while i < len(body):
        j = body.find(key, i)
        if j == -1:
            out.append(body[i:])
            break
        out.append(body[i:j])
        # \abbreviations{Title}{Body}
        open_title = j + len(key) - 1
        title, p1 = _extract_braced(body, open_title)
        if p1 >= len(body) or body[p1] != "{":
            raise ValueError("abbreviations: missing second argument")
        inner, p2 = _extract_braced(body, p1)
        out.append(f"\\section*{{{title}}}\n\n{inner}")
        i = p2
    return "".join(out)


def remove_commented_supplementary_templates(text: str) -> str:
    """Drop MDPI example lines like `% \\supplementary{...}` so they are not parsed by replace_command."""
    out: list[str] = []
    for line in text.splitlines(keepends=True):
        if re.search(r"^\s*%.*\\supplementary\{", line):
            continue
        out.append(line)
    return "".join(out)


def strip_mdpi_comments_boilerplate(text: str) -> str:
    """Remove leftover MDPI optional-supplement boilerplate that references undefined macros."""
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    for line in lines:
        if "\\linksupplementary" in line:
            continue
        low = line.lstrip()
        if low.startswith("% Only for journal Methods") or low.startswith("% Only for journal Hardware"):
            continue
        if low.startswith("% Only used for preprtints:"):
            continue
        if low.startswith("% If you wish to submit a video article"):
            continue
        if low.startswith("% \\section*{Supplementary Materials}"):
            continue
        if "posted on \\href{https://www.preprints.org/}" in line:
            continue
        out.append(line)
    return "".join(out)


def transform_mdpi_body(raw: str) -> str:
    text = remove_commented_supplementary_templates(raw)
    text = re.sub(r"\\begin\{linenomath\}\s*", "", text)
    text = re.sub(r"\\end\{linenomath\}\s*", "", text)

    text = re.sub(r"\\appendixtitles\{[^}]*\}\s*", "", text)
    text = re.sub(r"\\appendixstart\s*", "", text)

    text = re.sub(
        r"\\section\[[^\]]*\]\{([^}]*)\}",
        r"\\section{\1}",
        text,
    )
    text = re.sub(
        r"\\subsection\[[^\]]*\]\{([^}]*)\}",
        r"\\subsection{\1}",
        text,
    )

    # Order matters: longest / most specific first where needed
    text = replace_abbreviations_mdpi(text)
    text = replace_command(text, "supplementary", "\\section*{Supplementary Materials}\n\n")
    text = replace_command(text, "authorcontributions", "\\section*{Author Contributions}\n\n")
    text = replace_command(text, "funding", "\\paragraph{Funding.}\n")
    text = replace_command(text, "institutionalreview", "\\paragraph{Institutional Review Board Statement.}\n")
    text = replace_command(text, "informedconsent", "\\paragraph{Informed Consent Statement.}\n")
    text = replace_command(text, "dataavailability", "\\paragraph{Data Availability Statement.}\n")
    text = replace_command(text, "acknowledgments", "\\section*{Acknowledgments}\n\n")
    text = replace_command(text, "conflictsofinterest", "\\paragraph{Conflicts of Interest.}\n")
    text = strip_mdpi_comments_boilerplate(text)
    return text


def main() -> None:
    src = SRC.read_text(encoding="utf-8")
    m_doc = re.search(r"(?m)^\\begin\{document\}\s*$", src)
    if not m_doc:
        raise RuntimeError("Could not find literal \\\\begin{document} line")
    b0 = m_doc.end()
    b1 = src.index(r"\begin{adjustwidth}", b0)
    body = src[b0:b1].strip()

    body = transform_mdpi_body(body)

    OUT.write_text(PREAMBLE + "\n" + body + "\n" + POSTAMBLE, encoding="utf-8")
    print(f"Wrote {OUT.relative_to(DRAFT.parent)} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
