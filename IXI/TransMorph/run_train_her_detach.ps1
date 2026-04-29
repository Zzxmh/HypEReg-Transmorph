# Opens a separate CMD window for IXI TransMorph+HER training (batch_size in train_TransMorph_her.py).
# Closing Cursor/SSH does not close that window; remote SSH may still end child processes unless using mosh/task.

$here = $PSScriptRoot
$ixi = Split-Path $here -Parent
$repo = Split-Path $ixi -Parent
$py = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Error "Venv not found: $py"
    exit 1
}
$train = Join-Path $here "train_TransMorph_her.py"
if (-not (Test-Path $train)) {
    Write-Error "Training script not found: $train"
    exit 1
}
$cmdLine = "cd /d `"$here`" && `"$py`" -u train_TransMorph_her.py"
Start-Process -FilePath "cmd.exe" -ArgumentList "/k", $cmdLine -WindowStyle Normal
Write-Host "Started detached window: TransMorph+HER IXI (train batch_size=2 in script)."
