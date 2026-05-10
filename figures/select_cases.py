"""
Select 3 representative IXI subjects for the 3x3 deformation grid figure.
Criteria:
  Case A - High contrast: TransMorph has large non-pos-J, HypEReg is near zero
  Case B - Typical/median case
  Case C - Case with high SDlogJ ratio (worst distribution tails in TransMorph)
"""
import csv, os, glob
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
IXI_DATA = REPO_ROOT / "IXI_data" / "Test"

# ── Check data availability ──────────────────────────────────────────────────
pkls = sorted(glob.glob(str(IXI_DATA / "*.pkl")))
print(f"IXI Test PKL files found: {len(pkls)}")
if pkls:
    print("  First few:", [os.path.basename(p) for p in pkls[:5]])

# ── Load HypEReg per-subject Jacobian stats ───────────────────────────────────
her_jac = {}
with open(REPO_ROOT / "IXI/Results/comprehensive/HER_dsc0743/jacobian.csv") as f:
    for row in csv.DictReader(f):
        her_jac[row["subject"]] = {
            "non_jac": float(row["non_jac_frac"]),
            "SDlogJ": float(row["SDlogJ"]),
        }

# ── Load TransMorph per-subject non_jec from its CSV ─────────────────────────
tm_csv = REPO_ROOT / "IXI/Results/TransMorph_ncc_1_diffusion_1.csv"
tm_nonjec = {}
with open(tm_csv) as f:
    reader = csv.reader(f)
    next(reader)  # skip header row 1 (model name)
    header = next(reader)  # column names
    nonjec_idx = -1  # last column is non_jec in these CSVs
    for row in reader:
        if len(row) < 2:
            continue
        subj = row[0]
        try:
            tm_nonjec[subj] = float(row[nonjec_idx])
        except (ValueError, IndexError):
            pass

# ── Build contrast table ──────────────────────────────────────────────────────
common = sorted(set(her_jac.keys()) & set(tm_nonjec.keys()))
print(f"\nCommon subjects: {len(common)}")

rows = []
for s in common:
    her_nj = her_jac[s]["non_jac"]
    tm_nj = tm_nonjec[s]
    her_sdlogj = her_jac[s]["SDlogJ"]
    # Ratio: how many times worse is TransMorph
    ratio = tm_nj / (her_nj + 1e-9)
    rows.append((s, her_nj, tm_nj, ratio, her_sdlogj))

# Sort by ratio descending
rows.sort(key=lambda r: r[3], reverse=True)

print("\nTop 10 subjects by TransMorph/HypEReg non-pos-J ratio:")
for r in rows[:10]:
    print(f"  {r[0]:6s}  HER={r[1]:.6f}  TM={r[2]:.6f}  ratio={r[3]:.1f}  SDlogJ={r[4]:.4f}")

# ── Select 3 cases ────────────────────────────────────────────────────────────
# Case A: highest ratio (most dramatic contrast)
case_a = rows[0][0]

# Case B: median case by SDlogJ
all_sdlogj = sorted(rows, key=lambda r: r[4])
median_idx = len(all_sdlogj) // 2
case_b = all_sdlogj[median_idx][0]

# Case C: second highest ratio but different from A
case_c = rows[2][0]  # skip row[1] to ensure variety

print(f"\nSelected subjects:")
print(f"  Case A (max contrast):  {case_a}")
print(f"  Case B (median SDlogJ): {case_b}")
print(f"  Case C (3rd contrast):  {case_c}")

print(f"\nThese correspond to PKL files:")
for c in [case_a, case_b, case_c]:
    # PKL files are named like IXI002-Guys-0828-T1_0.pkl
    # subject names are p_0, p_1, etc. (index in test set)
    print(f"  {c} -> index {c.replace('p_', '')}")
