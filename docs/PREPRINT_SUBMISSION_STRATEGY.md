# Preprint and Journal Submission Strategy

This note documents whether to post the manuscript to arXiv before formal submission.

## 1) Policy baseline (MDPI Journal of Imaging)

According to the journal instructions:
- preprints are accepted (posting a preprint does not invalidate submission),
- MDPI also supports posting via Preprints.org after submission,
- preprint posting is independent from peer review.

Practical implication: **preprint posting is allowed** for this manuscript.

## 2) Recommended timing for this project

Recommended default:
1. submit to *Journal of Imaging* first,
2. post preprint the same day (or within 24h) using the exact submitted version.

Why this default works well:
- preserves priority claim and citation visibility,
- avoids version mismatch confusion during active peer review,
- keeps rebuttal edits and public versions easier to map.

## 3) If posting before submission

This is still acceptable, but enforce:
- frozen version tag (`v1`) with commit hash,
- title/abstract/metrics aligned with journal draft,
- explicit scope statement (IXI protocol),
- changelog policy for later revisions (`v2`, `v3`).

## 4) Version-control policy for manuscript revisions

For each preprint update:
- include a concise changelog section (what changed from previous version),
- keep old versions discoverable,
- map to repository release tags.

Suggested mapping:
- `paper-v1-submitted` -> initial journal submission / preprint v1
- `paper-v2-minor-revision` -> revised manuscript / preprint v2
- `paper-v3-accepted` -> accepted manuscript / camera-ready supplementary links

## 5) Suggested preprint disclosure sentence

Use in cover letter or repository:

> This preprint corresponds to a manuscript submitted to *Journal of Imaging*.  
> The current version is aligned with the submitted manuscript and may be updated to reflect peer-review revisions.

## 6) Decision for this release cycle

Given current state (active revision and supplementary/repository cleanup), the best strategy is:
- finalize release cleanup + supplementary index first,
- submit to journal,
- immediately post arXiv/Preprints version synchronized to the submitted files.
