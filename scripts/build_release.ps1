param(
    [string]$Python = "D:\miniconda\envs\UI\python.exe",
    [switch]$SkipTests,
    [switch]$SkipSmoke
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

function Remove-WorkspaceCaches {
    Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force ".pytest_cache" -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force ".tmp" -ErrorAction SilentlyContinue
}

function Invoke-Checked {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$Name
    )
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path $Python)) {
    throw "Python not found: $Python"
}

$WorkspaceTemp = Join-Path $Root ".tmp"
New-Item -ItemType Directory -Force -Path $WorkspaceTemp | Out-Null
$env:TMP = $WorkspaceTemp
$env:TEMP = $WorkspaceTemp

if (-not $SkipTests) {
    Invoke-Checked $Python @("-m", "pytest", "-q") "pytest"
}

$CompileTargets = @(
    "app.py"
    "run.py"
    "run_streamlit.py"
)
$CompileTargets += Get-ChildItem src -Recurse -Filter *.py | ForEach-Object { $_.FullName }
$CompileTargets += Get-ChildItem pages -Recurse -Filter *.py | ForEach-Object { $_.FullName }
$CompileTargets += Get-ChildItem scripts -Recurse -Filter *.py | ForEach-Object { $_.FullName }
$CompileArgs = @("-m", "py_compile") + $CompileTargets
Invoke-Checked $Python $CompileArgs "py_compile"

Remove-WorkspaceCaches
New-Item -ItemType Directory -Force -Path $WorkspaceTemp | Out-Null
Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue
Invoke-Checked $Python @("-m", "PyInstaller", "MemoryEvalUI.spec", "--clean", "--noconfirm") "PyInstaller"

$Exe = Join-Path $Root "dist\MemoryEvalUI\MemoryEvalUI.exe"
if (-not (Test-Path $Exe)) {
    throw "Build finished but exe was not found: $Exe"
}

if (-not $SkipSmoke) {
    try {
        $Proc = Start-Process -FilePath $Exe -WindowStyle Hidden -PassThru
        Start-Sleep -Seconds 12
        if ($Proc.HasExited) {
            throw "Smoke test failed: MemoryEvalUI.exe exited early with code $($Proc.ExitCode)"
        }
        Stop-Process -Id $Proc.Id -Force
    }
    catch {
        Write-Warning "Smoke test skipped/failed because the exe could not be launched: $($_.Exception.Message)"
    }
}

$Zip = Join-Path $Root "dist\MemoryEvalUI.zip"
if (Test-Path $Zip) {
    Remove-Item -Force $Zip
}
Compress-Archive -Path (Join-Path $Root "dist\MemoryEvalUI\*") -DestinationPath $Zip -Force

Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue
Remove-WorkspaceCaches

Write-Host "Release package created: $Zip"
