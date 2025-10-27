# Windows PowerShell runner for experiments
param(
    [string]$PerseusDir = "index",
    [string]$IndexDir = "index"
)

New-Item -ItemType Directory -Force -Path "output" | Out-Null

function Run-Step {
    param(
        [string]$Title,
        [string[]]$PyArgs,
        [string]$OutFile
    )
    Write-Host "Running $Title" -ForegroundColor Cyan
    try {
        & $py @PyArgs *>&1 | Tee-Object -FilePath $OutFile
    } catch {
        Write-Warning "Step '$Title' failed: $($_.Exception.Message)"
    }
}

$py = "python"
# Prefer project venv Python if available
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPy = Join-Path $scriptDir ".venv\\Scripts\\python.exe"
if (Test-Path $venvPy) { $py = $venvPy }

# Ensure all relative paths resolve from the repository root (this script's folder)
Set-Location $scriptDir

# Validate data root: prefer provided; else fallback to existing folder
if (-not (Test-Path $PerseusDir)) {
    if (Test-Path "index") { $PerseusDir = "index" }
    elseif (Test-Path "data") { $PerseusDir = "data" }
    else { Write-Warning "Neither '$PerseusDir' nor default 'index'/'data' exist. Please set -PerseusDir." }
}

# Detect if raw per-host time-series data exists (required by csr/multi_pred/lstm/patchTST/hybrid)
$hasPerHostData = (Test-Path (Join-Path $PerseusDir 'cluster_A')) -or (Test-Path (Join-Path $PerseusDir 'cluster_B'))

if ($hasPerHostData) {
    Run-Step -Title "csr.py (all_drive_info)" -PyArgs @("scripts/csr.py","-p",$PerseusDir,"-i", (Join-Path $IndexDir 'all_drive_info.csv')) -OutFile "output/csr_A.out"
    Run-Step -Title "csr.py (all_drive_info)" -PyArgs @("scripts/csr.py","-p",$PerseusDir,"-i", (Join-Path $IndexDir 'all_drive_info.csv')) -OutFile "output/csr_B.out"

    Run-Step -Title "multi_pred.py (all_drive_info)" -PyArgs @("scripts/multi_pred.py","-p",$PerseusDir,"-i", (Join-Path $IndexDir 'all_drive_info.csv')) -OutFile "output/multi_pred_A.out"
    Run-Step -Title "multi_pred.py (all_drive_info)" -PyArgs @("scripts/multi_pred.py","-p",$PerseusDir,"-i", (Join-Path $IndexDir 'all_drive_info.csv')) -OutFile "output/multi_pred_B.out"

    Run-Step -Title "lstm.py for cluster_A" -PyArgs @("scripts/lstm.py","-p",$PerseusDir,"-i", (Join-Path $IndexDir 'all_drive_info.csv'),"-t","cluster_A") -OutFile "output/lstm.out"

    Run-Step -Title "patchTST.py for cluster_A" -PyArgs @("scripts/patchTST.py","-p",$PerseusDir,"-i", (Join-Path $IndexDir 'all_drive_info.csv'),"-t","cluster_A") -OutFile "output/patchTST.out"
} else {
    Write-Host "Skipping csr/multi_pred/lstm/patchTST: no per-host raw time-series found under '$PerseusDir' (expected cluster_* folders)." -ForegroundColor Yellow
}

if ($env:OPENAI_API_KEY) {
    Run-Step -Title "GPT-4.py for index A" -PyArgs @("scripts/GPT-4.py","-p",$PerseusDir,"-i", (Join-Path $IndexDir 'A_index.csv')) -OutFile "output/GPT-4_A.out"
    Run-Step -Title "GPT-4.py for index B" -PyArgs @("scripts/GPT-4.py","-p",$PerseusDir,"-i", (Join-Path $IndexDir 'B_index.csv')) -OutFile "output/GPT-4_B.out"
} else {
    Write-Host "Skipping GPT-4 steps: OPENAI_API_KEY not set." -ForegroundColor Yellow
}

Run-Step -Title "iforest.py for index A" -PyArgs @("scripts/iforest.py","-p",".","-i", (Join-Path $IndexDir 'A_index.csv')) -OutFile "output/iforest_A.out"
Run-Step -Title "iforest.py for index B" -PyArgs @("scripts/iforest.py","-p",".","-i", (Join-Path $IndexDir 'B_index.csv')) -OutFile "output/iforest_B.out"

Run-Step -Title "svm.py for index A" -PyArgs @("scripts/svm.py","-p",".","-i", (Join-Path $IndexDir 'A_index.csv')) -OutFile "output/svm_A.out"
Run-Step -Title "svm.py for index B" -PyArgs @("scripts/svm.py","-p",".","-i", (Join-Path $IndexDir 'B_index.csv')) -OutFile "output/svm_B.out"

# Hybrid CNN–BiLSTM–Self-Attention (train on A, test on B)
if ($hasPerHostData) {
    Run-Step -Title "Hybrid CNN-BiLSTM-Attention A->B" -PyArgs @("scripts/hybrid_cnn_rnn_attention.py","-p",$PerseusDir,"-i", (Join-Path $IndexDir 'all_drive_info.csv'),"--train_cluster","cluster_A","--test_cluster","cluster_B","-o","output") -OutFile "output/hybrid.out"
} else {
    Write-Host "Skipping Hybrid model: no per-host raw time-series found under '$PerseusDir'." -ForegroundColor Yellow
}

Write-Host "Compressing outputs to output.zip" -ForegroundColor Green
if (Test-Path output.zip) { Remove-Item output.zip -Force }
Compress-Archive -Path .\output\* -DestinationPath .\output.zip

Write-Host "Done" -ForegroundColor Green
