# ============================================================
# setup_bot.ps1
# Повний автоматичний скрипт розгортання ai_scalper_bot на новому ПК
# ============================================================

Write-Host "`n=== AI Scalper Bot — Автоматичне встановлення середовища ===" -ForegroundColor Cyan

# ------------------------------------------------------------
# 0. Дозволяємо виконання скриптів (лише в рамках цієї сесії)
# ------------------------------------------------------------
Write-Host "Тимчасово дозволяю виконання скриптів..." -ForegroundColor Yellow
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force

# ------------------------------------------------------------
# 1. Перехід в директорію бота
# ------------------------------------------------------------
$BotPath = "C:\ai_scalper_bot"

if (-Not (Test-Path $BotPath)) {
    Write-Host "❌ Помилка: Папка $BotPath не знайдена!" -ForegroundColor Red
    exit
}

Set-Location $BotPath
Write-Host "Перехід у $BotPath" -ForegroundColor Green

# ------------------------------------------------------------
# 2. Перевірка Python
# ------------------------------------------------------------
Write-Host "Перевіряю Python..." -ForegroundColor Yellow

$pinfo = & python -V 2>$null

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Python не знайдено. Встанови Python 3.12.x перед запуском!" -ForegroundColor Red
    exit
}

Write-Host "Python знайдено: $pinfo" -ForegroundColor Green

if ($pinfo -notmatch "3\.12") {
    Write-Host "❌ Потрібен Python 3.12.x (Torch не працює на 3.13–3.14)" -ForegroundColor Red
    exit
}

# ------------------------------------------------------------
# 3. Створення або відновлення venv
# ------------------------------------------------------------
$VenvPath = "$BotPath\venv"
$ActivatePS1 = "$VenvPath\Scripts\Activate.ps1"
$ActivateBAT = "$VenvPath\Scripts\activate.bat"
$PythonVenv = "$VenvPath\Scripts\python.exe"

if (-Not (Test-Path $PythonVenv)) {
    Write-Host "Створюю venv..." -ForegroundColor Yellow
    python -m venv venv
} else {
    Write-Host "Venv знайдено." -ForegroundColor Green
}

# ------------------------------------------------------------
# 4. Активація venv через BAT (обхід ExecutionPolicy)
# ------------------------------------------------------------
Write-Host "Активую venv..." -ForegroundColor Yellow
cmd /c $ActivateBAT

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Не вдалося активувати venv!" -ForegroundColor Red
    exit
}

# ------------------------------------------------------------
# 5. Перевірка Python у venv
# ------------------------------------------------------------
Write-Host "Перевіряю Python у venv..." -ForegroundColor Yellow
$ver = & $PythonVenv -V

Write-Host "Python venv OK: $ver" -ForegroundColor Green

# ------------------------------------------------------------
# 6. Оновлення pip
# ------------------------------------------------------------
Write-Host "Оновлюю pip..." -ForegroundColor Yellow
& $PythonVenv -m pip install --upgrade pip setuptools wheel

# ------------------------------------------------------------
# 7. Встановлення основних пакетів
# ------------------------------------------------------------
Write-Host "Встановлюю основні залежності..." -ForegroundColor Cyan

$packages = @(
    "PyYAML>=6.0",
    "websockets>=11.0",
    "python-dotenv>=1.0",
    "aiohttp>=3.8",
    "numpy>=1.25",
    "pandas>=2.2",
    "python-binance>=1.0",
    "psutil",
    "pydantic",
    "requests",
    "xgboost"
)

foreach ($pkg in $packages) {
    Write-Host "⇢ $pkg" -ForegroundColor Gray
    & $PythonVenv -m pip install $pkg
}

# ------------------------------------------------------------
# 8. Установка Torch (CPU)
# ------------------------------------------------------------
Write-Host "Встановлюю Torch CPU..." -ForegroundColor Magenta

& $PythonVenv -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# ------------------------------------------------------------
# 9. Фінальна перевірка
# ------------------------------------------------------------
Write-Host "`nПеревіряю пакети..." -ForegroundColor Yellow

$tests = @(
    "numpy",
    "pandas",
    "torch",
    "aiohttp",
    "websockets",
    "xgboost",
    "pydantic",
    "yaml"
)

foreach ($mod in $tests) {
    Write-Host "→ Перевірка $mod ..." -ForegroundColor Gray
    & $PythonVenv - <<EOF
import $mod
print("$mod OK")
EOF
}

Write-Host "`n=== СИСТЕМА ПОВНІСТЮ ГОТОВА ДО ЗАПУСКУ БОТА ===" -ForegroundColor Green
Write-Host "Запуск бота: python -m bot.market_data.ws_manager" -ForegroundColor Cyan
