# Full OASIS pipeline: export half-res flows -> eval_oasis -> paired stats.
# Run from any directory; resolves repository root as parent of OASIS/.
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$models = @(
    "transmorph_her_oasis",
    "transmorph_unsup_oasis",
    "transmorph_dsc857_oasis",
    "voxelmorph_1_oasis",
    "midir_oasis",
    "cyclemorph_oasis",
    "syn_oasis",
    "affine_oasis"
)

foreach ($m in $models) {
    Write-Host "=== OASIS export: $m ===" -ForegroundColor Cyan
    python OASIS/export_displacements.py --model-id $m
    if ($LASTEXITCODE -ne 0) { throw "export_displacements failed for $m" }
    Write-Host "=== OASIS eval: $m ===" -ForegroundColor Cyan
    python OASIS/eval_oasis.py --models $m
    if ($LASTEXITCODE -ne 0) { throw "eval_oasis failed for $m" }
}

Write-Host "=== OASIS Wilcoxon / BH-FDR ===" -ForegroundColor Cyan
python OASIS/oasis_run_stats.py
if ($LASTEXITCODE -ne 0) { throw "oasis_run_stats failed" }

Write-Host "Done. Outputs: OASIS/Eval_Results/<model_id>/ and OASIS/Eval_Results/_stats/" -ForegroundColor Green
