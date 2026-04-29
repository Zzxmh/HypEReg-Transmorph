# Monitor OASIS TransMorph+HER training status from logfile and process table.
# Usage:
#   .\OASIS\monitor_train_her.ps1
#   .\OASIS\monitor_train_her.ps1 -IntervalSec 15 -WarnStaleSec 180
#   .\OASIS\monitor_train_her.ps1 -LogFile "OASIS\TransMorph\logs\TransMorph_OASIS_HER_ncc_1.0_grad_1.0_her_1.0_a0_b0.02_g20\logfile.log"

param(
  [int]$IntervalSec = 20,
  [int]$WarnStaleSec = 300,
  [string]$TrainScriptName = "train_TransMorph_her.py",
  [string]$LogFile = ""
)

function Resolve-RepoRoot {
  return Split-Path $PSScriptRoot -Parent
}

function Resolve-LogFile([string]$repo, [string]$manualPath) {
  if ($manualPath -and $manualPath.Trim().Length -gt 0) {
    if ([System.IO.Path]::IsPathRooted($manualPath)) {
      return $manualPath
    }
    return Join-Path $repo $manualPath
  }

  $logsRoot = Join-Path $repo "OASIS\TransMorph\logs"
  if (-not (Test-Path $logsRoot)) {
    return $null
  }

  $dirs = Get-ChildItem -Path $logsRoot -Directory -ErrorAction SilentlyContinue |
          Sort-Object LastWriteTime -Descending
  foreach ($d in $dirs) {
    $candidate = Join-Path $d.FullName "logfile.log"
    if (Test-Path $candidate) {
      return $candidate
    }
  }
  return $null
}

function Get-LatestIterLine([string[]]$tailLines) {
  for ($i = $tailLines.Length - 1; $i -ge 0; $i--) {
    if ($tailLines[$i] -match "^Iter\s+\d+\s+of\s+\d+\s+loss") {
      return $tailLines[$i]
    }
  }
  return $null
}

function Get-LatestEpochLine([string[]]$tailLines) {
  for ($i = $tailLines.Length - 1; $i -ge 0; $i--) {
    if ($tailLines[$i] -match "^Epoch\s+\d+\s+loss") {
      return $tailLines[$i]
    }
  }
  return $null
}

$repo = Resolve-RepoRoot
Set-Location $repo

$resolvedLog = Resolve-LogFile -repo $repo -manualPath $LogFile

Write-Host "Monitoring OASIS TransMorph+HER training..."
Write-Host "Repo: $repo"
if ($resolvedLog) {
  Write-Host "Log : $resolvedLog"
} else {
  Write-Host "Log : (not found yet, waiting for logfile creation)"
}
Write-Host "Interval=${IntervalSec}s, stale_warn=${WarnStaleSec}s"
Write-Host "Press Ctrl+C to stop."

while ($true) {
  $now = Get-Date
  $ts = $now.ToString("yyyy-MM-dd HH:mm:ss")
  Write-Host ""
  Write-Host "[$ts] ------------------------------"

  # Process status
  $procs = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
           Where-Object { $_.Name -like "python*.exe" -and $_.CommandLine -match [regex]::Escape($TrainScriptName) }
  if ($procs) {
    foreach ($p in $procs) {
      $p2 = Get-Process -Id $p.ProcessId -ErrorAction SilentlyContinue
      if ($p2) {
        $cpu = [Math]::Round($p2.CPU, 1)
        $wsMB = [Math]::Round($p2.WorkingSet64 / 1MB, 0)
        Write-Host "process: RUNNING  pid=$($p.ProcessId)  cpu_s=$cpu  ws_mb=$wsMB"
      } else {
        Write-Host "process: RUNNING  pid=$($p.ProcessId)"
      }
    }
  } else {
    Write-Host "process: NOT FOUND for $TrainScriptName"
  }

  # Log status
  if (-not $resolvedLog) {
    $resolvedLog = Resolve-LogFile -repo $repo -manualPath $LogFile
  }

  if ($resolvedLog -and (Test-Path $resolvedLog)) {
    $logInfo = Get-Item $resolvedLog -ErrorAction SilentlyContinue
    $ageSec = [Math]::Round(($now - $logInfo.LastWriteTime).TotalSeconds, 1)
    $sizeKB = [Math]::Round($logInfo.Length / 1KB, 1)
    $staleTag = if ($ageSec -gt $WarnStaleSec) { "STALE" } else { "fresh" }
    Write-Host "log: $staleTag  age_s=$ageSec  size_kb=$sizeKB"

    $tail = Get-Content $resolvedLog -Tail 40 -ErrorAction SilentlyContinue
    $iterLine = Get-LatestIterLine -tailLines $tail
    $epochLine = Get-LatestEpochLine -tailLines $tail

    if ($epochLine) { Write-Host "latest_epoch: $epochLine" }
    if ($iterLine) { Write-Host "latest_iter : $iterLine" }

    Write-Host "tail:"
    $tail | Select-Object -Last 8 | ForEach-Object { Write-Host "  $_" }
  } else {
    Write-Host "log: logfile.log not found yet"
  }

  Start-Sleep -Seconds $IntervalSec
}
