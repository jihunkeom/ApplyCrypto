<#
.SYNOPSIS
    This script downloads Python wheel packages for production and development dependencies.
.DESCRIPTION
    This script ensures a 'wheels' directory exists, then uses pip to download all
    dependencies listed in 'requirements.txt' and 'requirements-dev.txt' into that directory.
    This is useful for creating an offline package cache.
#>


$wheelsDir = "./wheels"

# 1. Prepare Directory: Ensure wheels directory exists.
if (-not (Test-Path $wheelsDir)) {
    New-Item -ItemType Directory -Path $wheelsDir -Force | Out-Null
    Write-Host "Created directory: $wheelsDir"
}
else {
    Write-Host "Directory already exists: $wheelsDir"
}

# 1.5. Download pip infrastructure packages: wheel and setuptools
Write-Host "Downloading essential pip infrastructure packages (wheel, setuptools)..."
pip wheel wheel setuptools -w $wheelsDir
if ($LASTEXITCODE -eq 0) {
    Write-Host "Essential packages downloaded successfully." -ForegroundColor Green
}
else {
    Write-Host "ERROR: Failed to download essential packages. Exit code: $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}


# 2. Download Production Dependencies: Run pip wheel -r requirements.txt -w ./wheels
Write-Host "Downloading production dependencies..."
pip wheel -r requirements.txt -w $wheelsDir
if ($LASTEXITCODE -eq 0) {
    Write-Host "Production dependencies downloaded successfully." -ForegroundColor Green
}
else {
    Write-Host "ERROR: Failed to download production dependencies. Exit code: $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

# 3. Download Development Dependencies: Run pip wheel -r requirements-dev.txt -w ./wheels
Write-Host "Downloading development dependencies..."
pip wheel -r requirements-dev.txt -w $wheelsDir
if ($LASTEXITCODE -eq 0) {
    Write-Host "Development dependencies downloaded successfully." -ForegroundColor Green
}
else {
    Write-Host "ERROR: Failed to download development dependencies. Exit code: $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
