# Git Diff 생성 및 적용 가이드 (Windows PowerShell)

Windows PowerShell 환경에서 Git Diff를 생성하고 적용할 때 발생할 수 있는 인코딩 문제와 올바른 사용법을 설명합니다.

## 1. Diff 생성 (Generating Diff)

PowerShell에서 리다이렉션 연산자(`>`)를 사용하여 diff 파일을 생성하면, 기본적으로 **UTF-16LE** 인코딩으로 저장됩니다. `git apply`는 이를 인식하지 못해 `error: No valid patches in input` 오류가 발생할 수 있습니다.

### 올바른 생성 방법

다음 두 가지 방법 중 하나를 사용하여 **UTF-8** 또는 **ASCII**로 강제 저장해야 합니다.

#### 방법 1: `cmd` 셸 사용 (권장)
PowerShell 내에서 `cmd`를 호출하여 리다이렉션을 수행합니다.
```powershell
cmd /c "git diff [COMMIT_A] [COMMIT_B] > change.diff"
```
예시:
```powershell
cmd /c "git diff 293a1f6 8137c8c > change_1224.diff"
```

#### 방법 2: `Set-Content` 사용
PowerShell 파이프라인과 `Set-Content`를 사용하여 인코딩을 명시합니다.
```powershell
git diff [COMMIT_A] [COMMIT_B] | Set-Content -Encoding utf8 change.diff
```

---

## 2. Diff 적용 (Applying Diff)

생성된 patch 파일을 적용할 때는 `git apply` 명령어를 사용합니다.

### 기본 적용
```powershell
git apply change.diff
```

### 공백 무시 및 오류 완화 옵션
충돌이나 공백 문제로 적용이 안 될 경우 다음 옵션들을 활용하세요.

```powershell
# 공백 변경 무시 및 줄바꿈 문자 자동 수정
git apply --ignore-space-change --ignore-whitespace --whitespace=fix change.diff
```

### 적용 전 테스트 (Dry Run)
실제로 적용하기 전에 오류가 없는지 미리 확인해볼 수 있습니다.
```powershell
git apply --check change.diff
```
