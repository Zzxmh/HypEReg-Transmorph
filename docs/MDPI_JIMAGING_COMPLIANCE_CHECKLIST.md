# MDPI Journal of Imaging Compliance Checklist

This checklist is tailored for submission of the HER-TransMorph manuscript to MDPI *Journal of Imaging*.

Primary policy sources:
- [Journal of Imaging Instructions for Authors](https://www.mdpi.com/journal/jimaging/instructions)
- [MDPI Layout Style Guide](https://www.mdpi.com/authors/layout)
- [MDPI Research and Publication Ethics](https://www.mdpi.com/ethics)

## A) Supplementary Materials Compliance

- [ ] Supplementary section exists in manuscript back matter (already present in `draft/template_ixi_only.tex`).
- [ ] Every supplementary item is named with `S` prefix (`Figure S1`, `Table S1`, `Table S2`, etc.).
- [ ] Every supplementary item is cited in the main text at least once.
- [ ] Supplementary section explicitly lists all items by name and title.
- [ ] If supplementary references are used, those references are also included in the main bibliography.
- [ ] Externally hosted supplementary files (if any) include repository URL and accessed date.

## B) Data Availability Statement (Required)

- [ ] Data Availability Statement is present (required by MDPI for all articles).
- [ ] Statement includes where core data can be accessed:
  - source dataset URL(s),
  - derived data/output location (repository + DOI/URL),
  - restrictions (if any) with explicit reason.
- [ ] If some data cannot be shared (size/privacy/license), limitations are stated clearly.
- [ ] Data statement in manuscript matches what is actually released.

## C) Code and Software Availability

- [ ] Code used for reported results is public or uploaded as supplementary software.
- [ ] Software versions are specified (framework and key libraries).
- [ ] Reproduction-critical scripts are identified in Methods/Supplementary.
- [ ] Public repository links are stable (tag/release/commit pinned).

## D) Reproducibility Metadata

- [ ] Minimal dataset needed to support central conclusions is available.
- [ ] Training/inference configs used for paper are included.
- [ ] Table and figure generation scripts are included.
- [ ] Artifact provenance is documented (which script generated which file).
- [ ] Checksums or hashes are provided for large downloadable assets (recommended).

## E) File Format and Archival Good Practice

- [ ] Prefer non-proprietary formats where possible (`.csv`, `.txt`, `.yaml`, `.json`, `.pdf`, `.png`).
- [ ] Large assets are deposited in a trusted repository with DOI when possible.
- [ ] No personal website links for critical artifacts.

## F) Preprint Policy (Relevant to submission timing)

- [ ] Confirmed: J. Imaging accepts manuscripts previously posted as preprints.
- [ ] If posting preprint, manuscript version and journal submission version are synchronized.
- [ ] Preprint text should not conflict with under-review claims/results.

## G) Final Pre-Submission Gate

- [ ] Manuscript supplementary citations and supplementary files are consistent.
- [ ] Data availability wording matches released repository state.
- [ ] Code availability wording matches released repository state.
- [ ] No local absolute machine paths appear in released metadata/logs.
- [ ] Release README points to supplementary index and reproduction commands.
