# Full Setup and Launch Script for Real-time Trading Dashboard
# Run this from the ai-crypto-trading-dashboard directory

param(
    [switch]$SkipDownload,
    [switch]$SkipTrain,
    [int]$Days = 90
)

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Real-time Crypto Trading Dashboard Setup" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check dependencies
Write-Host "Step 1: Checking dependencies..." -ForegroundColor Yellow

$required_packages = @("torch", "pandas", "streamlit", "plotly", "websocket-client")
$missing_packages = @()

foreach ($pkg in $required_packages) {
    try {
        $null = python -c "import $pkg" 2>&1
        Write-Host "  ✓ $pkg installed" -ForegroundColor Green
    }
    catch {
        $missing_packages += $pkg
        Write-Host "  ✗ $pkg missing" -ForegroundColor Red
    }
}

if ($missing_packages.Count -gt 0) {
    Write-Host ""
    Write-Host "Installing missing packages..." -ForegroundColor Yellow
    pip install websocket-client streamlit plotly requests
    Write-Host ""
}

# Step 2: Download data
if (-not $SkipDownload) {
    Write-Host "Step 2: Downloading $Days days of BTC data from Binance..." -ForegroundColor Yellow
    Write-Host "  (This will take 2-3 minutes)" -ForegroundColor Gray
    Write-Host ""
    
    python scripts/download_binance_historical.py --symbol BTCUSDT --days $Days --out data/btcusdt_1m.parquet
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "❌ Data download failed!" -ForegroundColor Red
        exit 1
    }
    
    Write-Host ""
}
else {
    Write-Host "Step 2: Skipping data download (--SkipDownload specified)" -ForegroundColor Gray
    Write-Host ""
}

# Check if data exists
if (-not (Test-Path "data/btcusdt_1m.parquet")) {
    Write-Host "❌ Data file not found: data/btcusdt_1m.parquet" -ForegroundColor Red
    Write-Host "   Run without --SkipDownload to download data" -ForegroundColor Yellow
    exit 1
}

# Step 3: Train model
if (-not $SkipTrain) {
    Write-Host "Step 3: Training model..." -ForegroundColor Yellow
    Write-Host "  (This will take 5-10 minutes)" -ForegroundColor Gray
    Write-Host ""
    
    python src/train.py --config configs/default.yaml
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "❌ Training failed!" -ForegroundColor Red
        exit 1
    }
    
    Write-Host ""
}
else {
    Write-Host "Step 3: Skipping training (--SkipTrain specified)" -ForegroundColor Gray
    Write-Host ""
}

# Check if model artifacts exist
$required_files = @(
    "artifacts/model.pt",
    "artifacts/scaler.pkl",
    "artifacts/meta.json"
)

$missing_files = @()
foreach ($file in $required_files) {
    if (-not (Test-Path $file)) {
        $missing_files += $file
    }
}

if ($missing_files.Count -gt 0) {
    Write-Host "❌ Missing model artifacts:" -ForegroundColor Red
    foreach ($file in $missing_files) {
        Write-Host "   - $file" -ForegroundColor Red
    }
    Write-Host "   Run without --SkipTrain to train the model" -ForegroundColor Yellow
    exit 1
}

# Step 4: Validate features
Write-Host "Step 4: Validating feature alignment..." -ForegroundColor Yellow
Write-Host ""

python scripts/validate_live_features.py --data data/btcusdt_1m.parquet

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "❌ Feature validation failed!" -ForegroundColor Red
    Write-Host "   Live features don't match training features." -ForegroundColor Red
    Write-Host "   This needs to be fixed before proceeding." -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Step 5: Launch dashboard
Write-Host "================================================" -ForegroundColor Green
Write-Host "  ✓ Setup Complete! Launching Dashboard..." -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Dashboard will open at: http://localhost:8501" -ForegroundColor Cyan
Write-Host ""
Write-Host "In the dashboard:" -ForegroundColor Yellow
Write-Host "  1. Click '▶️ Start' in the sidebar" -ForegroundColor Yellow
Write-Host "  2. Wait 1-2 minutes for initial data" -ForegroundColor Yellow
Write-Host "  3. Watch live signals appear!" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press Ctrl+C to stop the dashboard" -ForegroundColor Gray
Write-Host ""

Start-Sleep -Seconds 2

streamlit run dashboard/app.py

