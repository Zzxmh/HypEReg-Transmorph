# Opens a separate CMD window for IXI TransMorph training (batch_size in train_TransMorph.py).
# Closing Cursor/SSH does not close that window; remote SSH may still end child processes unless using mosh/task.

$here = $PSScriptRoot
$ixi = Split-Path $here -Parent
$repo = Split-Path $ixi -Parent
$py = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Error "Venv not found: $py"
    exit 1
}
$train = Join-Path $here "train_TransMorph.py"
$cmdLine = "cd /d `"$here`" && `"$py`" -u train_TransMorph.py"
Start-Process -FilePath "cmd.exe" -ArgumentList "/k", $cmdLine -WindowStyle Normal
Write-Host "Started detached window: TransMorph IXI (train batch_size=2 in script)."
