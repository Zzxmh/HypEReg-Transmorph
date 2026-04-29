<#
    Sequentially train the IXI atlas-to-image baselines on a single GPU:
      1) VoxelMorph (train_vxm.py)
      2) MIDIR      (train_MIDIR.py)

    Usage (from anywhere):
        powershell -ExecutionPolicy Bypass -File .\run_vxm_then_midir.ps1
        powershell -ExecutionPolicy Bypass -File .\run_vxm_then_midir.ps1 -Gpu 0
        powershell -ExecutionPolicy Bypass -File .\run_vxm_then_midir.ps1 -SkipVxm       # only MIDIR
        powershell -ExecutionPolicy Bypass -File .\run_vxm_then_midir.ps1 -SkipMidir     # only VoxelMorph
        powershell -ExecutionPolicy Bypass -File .\run_vxm_then_midir.ps1 -ResumeVxm    # VoxelMorph from best dsc*.pth.tar, then MIDIR
        powershell -ExecutionPolicy Bypass -File .\run_vxm_then_midir.ps1 -ResumeVxm -VxmCkpt ".\VoxelMorph\experiments\vxm_1_ncc_1_diffusion_1\dsc0.569.pth.tar"
        powershell -ExecutionPolicy Bypass -File .\run_vxm_then_midir.ps1 -ResumeVxm -SkipMidir  # only resume VoxelMorph

    Notes:
      - Each training runs in its own subfolder so logs / experiments are written
        relative to that method (e.g. VoxelMorph/experiments, MIDIR/experiments).
      - Uses $env:CUDA_VISIBLE_DEVICES to pin the visible GPU; inside the python
        scripts, GPU_iden is 0 so this script effectively chooses the physical GPU.
      - The second job only starts after the first one exits successfully, unless
        -ContinueOnError is given.
#>

[CmdletBinding()]
param(
    [int]$Gpu = 0,
    [switch]$SkipVxm,
    [switch]$SkipMidir,
    [switch]$ContinueOnError,
    [switch]$ResumeVxm,
    [string]$VxmCkpt = ''
)

$ErrorActionPreference = 'Stop'

# Resolve folders relative to this script so the runner works from any cwd.
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VxmDir    = Join-Path $ScriptDir 'VoxelMorph'
$MidirDir  = Join-Path $ScriptDir 'MIDIR'

$VxmScript   = Join-Path $VxmDir   'train_vxm.py'
$MidirScript = Join-Path $MidirDir 'train_MIDIR.py'

foreach ($p in @($VxmScript, $MidirScript)) {
    if (-not (Test-Path $p)) {
        throw "Training script not found: $p"
    }
}

# Prefer repo .venv (two levels up from Baseline_registration_methods = repo root).
$RepoRoot = Split-Path (Split-Path $ScriptDir)
$VenvPy = Join-Path $RepoRoot '.venv\Scripts\python.exe'
if (Test-Path $VenvPy) {
    $Python = (Resolve-Path $VenvPy).Path
} else {
    $Python = (Get-Command python -ErrorAction SilentlyContinue).Source
}
if (-not $Python) {
    throw "Could not find $VenvPy or 'python' on PATH. Activate your env first."
}

$env:CUDA_VISIBLE_DEVICES = "$Gpu"
$env:PYTHONUNBUFFERED = '1'

function Invoke-Training {
    param(
        [Parameter(Mandatory)] [string]$Name,
        [Parameter(Mandatory)] [string]$WorkDir,
        [Parameter(Mandatory)] [string]$ScriptPath,
        [string[]]$ScriptArgs = @()
    )

    $sep = ('=' * 72)
    Write-Host ""
    Write-Host $sep -ForegroundColor Cyan
    Write-Host ("[{0}] Starting {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Name) -ForegroundColor Cyan
    Write-Host ("  cwd   : {0}" -f $WorkDir)
    Write-Host ("  script: {0}" -f $ScriptPath)
    if ($ScriptArgs.Count -gt 0) { Write-Host ("  args  : {0}" -f ($ScriptArgs -join ' ')) }
    Write-Host ("  gpu   : CUDA_VISIBLE_DEVICES={0}" -f $env:CUDA_VISIBLE_DEVICES)
    Write-Host $sep -ForegroundColor Cyan

    $start = Get-Date
    Push-Location $WorkDir
    try {
        & $Python -u (Split-Path -Leaf $ScriptPath) @ScriptArgs
        $code = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
    $elapsed = (Get-Date) - $start
    Write-Host ("[{0}] {1} finished with exit code {2} in {3:hh\:mm\:ss}" `
        -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Name, $code, $elapsed) -ForegroundColor Yellow

    if ($code -ne 0) {
        if ($ContinueOnError) {
            Write-Warning "$Name failed (exit $code) but -ContinueOnError was set; continuing."
        } else {
            throw "$Name failed with exit code $code. Aborting sequence."
        }
    }
}

$globalStart = Get-Date

if (-not $SkipVxm) {
    $vxmArgs = @()
    if ($VxmCkpt -and $VxmCkpt.Trim().Length -gt 0) {
        $vxmArgs = @('--ckpt', $VxmCkpt)
    } elseif ($ResumeVxm) {
        $vxmArgs = @('--resume')
    }
    $vxmLabel = 'VoxelMorph (IXI atlas->image)'
    if ($vxmArgs.Count -gt 0) { $vxmLabel += ' [resume]' }
    Invoke-Training -Name $vxmLabel -WorkDir $VxmDir -ScriptPath $VxmScript -ScriptArgs $vxmArgs
} else {
    Write-Host "Skipping VoxelMorph (-SkipVxm)" -ForegroundColor DarkYellow
}

if (-not $SkipMidir) {
    Invoke-Training -Name 'MIDIR (IXI atlas->image)' -WorkDir $MidirDir -ScriptPath $MidirScript
} else {
    Write-Host "Skipping MIDIR (-SkipMidir)" -ForegroundColor DarkYellow
}

$totalElapsed = (Get-Date) - $globalStart
Write-Host ""
Write-Host ("All requested jobs done. Total elapsed: {0:hh\:mm\:ss}" -f $totalElapsed) -ForegroundColor Green
