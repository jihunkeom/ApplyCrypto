# 설정
$PYTHON_REQUIRED_VERSION = "3.13"
$WHEEL_PATH = "wheels"
$REQUIREMENTS_FILE = "requirements.txt"
$VENV_DIR = "venv"

function Confirm-Success {
    param(
        [string]$ErrorMessage
    )
    if ($LASTEXITCODE -ne 0) {
        Write-Error $ErrorMessage
        exit 1
    }
}

# Python 버전 확인
Write-Host "Python 버전 확인 중..."
try {
    $pythonVersionOutput = python --version 2>&1
    if ($pythonVersionOutput -match "Python\s+(\d+\.\d+)") {
        $currentVersion = $matches[1]
        if ($currentVersion -ne $PYTHON_REQUIRED_VERSION) {
            Write-Error "오류: Python 버전 $PYTHON_REQUIRED_VERSION 이(가) 필요하지만 $currentVersion 이(가) 발견되었습니다."
            exit 1
        }
        Write-Host "Python 버전 $currentVersion 확인됨."
    }
    else {
        throw "버전 파싱 실패"
    }
}
catch {
    Write-Error "오류: Python을 찾을 수 없거나 버전을 확인할 수 없습니다."
    exit 1
}

# Wheels 디렉토리 확인
if (-not (Test-Path -Path ".\$WHEEL_PATH" -PathType Container)) {
    Write-Error "오류: 현재 디렉토리에서 '$WHEEL_PATH' 디렉토리를 찾을 수 없습니다."
    Write-Host "스크립트를 실행하기 전에 wheels 아카이브 압축을 현재 디렉토리에 풀어주세요."
    exit 1
}
Write-Host "'$WHEEL_PATH' 디렉토리 확인됨."

# 실행 정책 확인 및 설정
$currentExecutionPolicy = Get-ExecutionPolicy -Scope CurrentUser

if ($currentExecutionPolicy -ne "RemoteSigned") {
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
    Write-Host "실행 정책 Remote Signed 설정됨."
}
else {
    Write-Host "실행 정책이 이미 RemoteSigned입니다."
}

# 가상 환경 생성 (존재하지 않을 경우)
$venvPath = $null
$possibleVenvDirs = @(".venv", "venv")

foreach ($dir in $possibleVenvDirs) {
    if (Test-Path -Path ".\$dir" -PathType Container) {
        $checkPath = Join-Path ".\$dir" "Scripts\Activate.ps1"
        if (Test-Path -Path $checkPath) {
            $venvPath = ".\$dir"
            Write-Host "$venvPath 에서 유효한 가상 환경 발견됨."
            break
        }
    }
}

if ($null -eq $venvPath) {
    $venvPath = ".\$VENV_DIR"
    python -m venv $venvPath
    Confirm-Success "오류: 가상 환경 생성 실패."
}

# 가상 환경 활성화
Write-Host "가상 환경 활성화 중..."
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
if (Test-Path -Path $activateScript) {
    . $activateScript
}
else {
    Write-Error "오류: 활성화 스크립트를 찾을 수 없음: $activateScript"
    exit 1
}

# 패키지 설치
Write-Host "'$WHEEL_PATH' 디렉토리에서 패키지 설치 중..."
$pipExe = Join-Path $venvPath "Scripts\pip.exe"

# 1. setuptools, wheel 설치
& $pipExe install --no-index --find-links=$WHEEL_PATH setuptools wheel
Confirm-Success "오류: setuptools 및 wheel 설치 실패."

# 2. 프로젝트를 편집 모드로 설치
& $pipExe install --no-index --find-links=$WHEEL_PATH --no-build-isolation -e .
Confirm-Success "오류: 프로젝트를 편집 모드로 설치 실패."

# 3. requirements.txt 패키지 설치
& $pipExe install --no-index --find-links=$WHEEL_PATH -r $REQUIREMENTS_FILE
Confirm-Success "오류: 패키지 설치 실패."

# applycrypto 명령어 확인
Write-Host "'applycrypto' 명령어 확인 중..."
if (Get-Command applycrypto -ErrorAction SilentlyContinue) {
    Write-Host "성공: 'applycrypto' 명령어 확인됨."
    applycrypto --help
}
else {
    Write-Error "오류: 가상 환경에서 'applycrypto' 명령어를 찾을 수 없습니다."
    exit 1
}

Write-Host "설정 완료."
