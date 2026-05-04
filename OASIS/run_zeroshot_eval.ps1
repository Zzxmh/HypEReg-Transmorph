# IXI -> OASIS zero-shot pipeline.
# For each IXI-trained model: export half-res displacement fields,
# compute the full metric panel, then run paired Wilcoxon/BH-FDR stats.
#
# Usage (from any directory):
#   powershell -File OASIS\run_zeroshot_eval.ps1
#
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$models = @(
    "transmorph_zs_oasis",
    "transmorphbayes_zs_oasis",
    "transmorph_her_zs_oasis",
    "cotr_zs_oasis",
    "nnformer_zs_oasis",
    "pvt_zs_oasis"
)

foreach ($m in $models) {
    Write-Host "=== [ZS] export: $m ===" -ForegroundColor Cyan
    python OASIS/export_displacements.py --model-id $m
    if ($LASTEXITCODE -ne 0) { throw "export_displacements failed for $m" }

    Write-Host "=== [ZS] eval: $m ===" -ForegroundColor Cyan
    python OASIS/eval_oasis.py --models $m
    if ($LASTEXITCODE -ne 0) { throw "eval_oasis failed for $m" }
}

Write-Host "=== [ZS] Wilcoxon / BH-FDR stats ===" -ForegroundColor Cyan
python OASIS/oasis_run_stats.py
if ($LASTEXITCODE -ne 0) { throw "oasis_run_stats failed" }

Write-Host "Done. Outputs: OASIS/Eval_Results/<model_id>/ and OASIS/Eval_Results/_stats/" -ForegroundColor Green
