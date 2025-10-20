# PowerShell script to launch the trading dashboard
# Run this from the ai-crypto-trading-dashboard directory

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Real-time Crypto Trading Dashboard Launcher" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Check if required files exist
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
    Write-Host "❌ Missing required files:" -ForegroundColor Red
    foreach ($file in $missing_files) {
        Write-Host "   - $file" -ForegroundColor Red
    }
    Write-Host ""
    Write-Host "Please train the model first:" -ForegroundColor Yellow
    Write-Host "   python src/train.py --config configs/default.yaml" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

Write-Host "✓ Model artifacts found" -ForegroundColor Green
Write-Host ""

# Check if streamlit is installed
try {
    $null = Get-Command streamlit -ErrorAction Stop
    Write-Host "✓ Streamlit installed" -ForegroundColor Green
} catch {
    Write-Host "❌ Streamlit not found" -ForegroundColor Red
    Write-Host ""
    Write-Host "Installing required packages..." -ForegroundColor Yellow
    pip install streamlit plotly websocket-client requests
    Write-Host ""
}

Write-Host ""
Write-Host "🚀 Launching dashboard..." -ForegroundColor Cyan
Write-Host ""
Write-Host "Dashboard will open in your browser at: http://localhost:8501" -ForegroundColor Yellow
Write-Host ""
Write-Host "To stop the dashboard, press Ctrl+C" -ForegroundColor Yellow
Write-Host ""

# Launch streamlit
streamlit run dashboard/app.py

