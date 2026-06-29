# ml-lab-curriculum — One-Time Setup Script
# Run this in your own PowerShell terminal (NOT inside the IDE sandbox)
# Usage: cd C:\Users\Lenovo\OneDrive\Desktop\NanoTab\ml-lab-curriculum
#        .\setup.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$BASE = $PSScriptRoot

function Log-Step { param($msg) Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Log-OK   { param($msg) Write-Host "    OK: $msg" -ForegroundColor Green }
function Log-Fail { param($msg) Write-Host "    FAIL: $msg" -ForegroundColor Red }

# ── STEP 2: Install uv if missing ─────────────────────────────────────────────
Log-Step "STEP 2 — Installing uv"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    # Refresh PATH
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","User") + ";" + $env:PATH
}
uv --version | Write-Host
Log-OK "uv ready"

# ── STEP 2b: Create venv with Python 3.11 ─────────────────────────────────────
Log-Step "STEP 2b — Creating .venv with Python 3.11"
Set-Location $BASE
uv venv --python 3.11
$PYTHON = "$BASE\.venv\Scripts\python.exe"
$PIP    = "$BASE\.venv\Scripts\pip.exe"
Log-OK ".venv created"

# ── STEP 3: Clone TFM-Playground ──────────────────────────────────────────────
Log-Step "STEP 3 — Cloning TFM-Playground"
if (-not (Test-Path "$BASE\TFM-Playground")) {
    git clone https://github.com/automl/TFM-Playground.git "$BASE\TFM-Playground"
} else { Write-Host "    Already cloned, skipping" }
& $PIP install -e "$BASE\TFM-Playground" --quiet
Log-OK "TFM-Playground installed"

# ── STEP 4: Clone TabICL ──────────────────────────────────────────────────────
Log-Step "STEP 4 — Cloning TabICL"
if (-not (Test-Path "$BASE\tabicl")) {
    git clone https://github.com/soda-inria/tabicl.git "$BASE\tabicl"
} else { Write-Host "    Already cloned, skipping" }
& $PIP install -e "$BASE\tabicl" --quiet
Log-OK "TabICL installed"

# ── STEP 5: Install additional dependencies ────────────────────────────────────
Log-Step "STEP 5 — Installing dependencies"
& $PIP install numpy pandas matplotlib seaborn scikit-learn jupyter notebook ipykernel torch torchvision h5py ruff --quiet
Log-OK "Dependencies installed"

# ── STEP 6: Download pretrained data dumps ────────────────────────────────────
Log-Step "STEP 6 — Downloading data dumps (this may take a while)"
$DATA = "$BASE\data"

$classFile = "$DATA\50x3_3_100k_classification.h5"
if (-not (Test-Path $classFile)) {
    Write-Host "    Downloading classification dump (~GB)..."
    Invoke-WebRequest -Uri "https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/50x3_3_100k_classification.h5" `
        -OutFile $classFile -UseBasicParsing
    Log-OK "Classification dump saved"
} else { Write-Host "    Classification dump already present, skipping" }

$regFile = "$DATA\50x3_1280k_regression.h5"
if (-not (Test-Path $regFile)) {
    Write-Host "    Downloading regression dump (~GB)..."
    Invoke-WebRequest -Uri "https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/50x3_1280k_regression.h5" `
        -OutFile $regFile -UseBasicParsing
    Log-OK "Regression dump saved"
} else { Write-Host "    Regression dump already present, skipping" }

# ── STEP 7: Verify classifier ─────────────────────────────────────────────────
Log-Step "STEP 7 — Verifying NanoTabPFNClassifier"
& $PYTHON -c @"
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import sys
sys.path.insert(0, r'$BASE\TFM-Playground')
from tfmplayground import NanoTabPFNClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
clf = NanoTabPFNClassifier()
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
probs = clf.predict_proba(X_test)
print('Accuracy:', accuracy_score(y_test, preds))
print('ROC AUC:', roc_auc_score(y_test, probs[:, 1]))
print('SUCCESS: environment is working correctly')
"@

# ── STEP 8: TabICL prior test ─────────────────────────────────────────────────
Log-Step "STEP 8 — Verifying TabICL prior"
& $PYTHON -c @"
import sys
sys.path.insert(0, r'$BASE\TFM-Playground')
from tfmplayground.external_priors import TabICLPriorDataLoader

prior = TabICLPriorDataLoader(
    num_steps=3, batch_size=2, num_datapoints_max=50,
    min_features=3, max_features=3, max_num_classes=3, device='cpu'
)
batch = next(iter(prior))
print('Prior batch keys:', batch.keys() if hasattr(batch, 'keys') else type(batch))
print('SUCCESS: TabICL prior is generating data correctly')
"@

# ── STEP 9: Smoke-test pretraining ────────────────────────────────────────────
Log-Step "STEP 9 — Pretraining smoke test (2 epochs, 5 steps)"
Set-Location "$BASE\TFM-Playground"
& $PYTHON pretrain_classification.py `
    --epochs 2 --steps 5 --batchsize 10 `
    --priordump "$DATA\50x3_3_100k_classification.h5"
Set-Location $BASE

# ── STEP 10: requirements.txt snapshot ────────────────────────────────────────
Log-Step "STEP 10 — Freezing requirements.txt"
& $PIP freeze | Out-File -FilePath "$BASE\requirements.txt" -Encoding utf8
Log-OK "requirements.txt written"

# ── STEP 11: Git init ─────────────────────────────────────────────────────────
Log-Step "STEP 11 — Initialising git repository"
Set-Location $BASE
git init
# Add a .gitignore first so we don't commit large data files or venv
Copy-Item "$BASE\.gitignore" "$BASE\.gitignore" -ErrorAction SilentlyContinue
git add --all
git commit -m "initial setup: TFM-Playground + TabICL prior + data dumps"
Log-OK "git repo initialised"

# ── FINAL: Print tree ─────────────────────────────────────────────────────────
Log-Step "FINAL — Folder structure"
Get-ChildItem $BASE -Recurse -Depth 3 |
    Where-Object { $_.FullName -notmatch '\\\.git\\|\\__pycache__\\|\\\.venv\\' } |
    Select-Object FullName

Write-Host "`n All steps complete!" -ForegroundColor Green
