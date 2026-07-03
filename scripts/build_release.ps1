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

function Invoke-Capture {
    param(
        [string]$FilePath,
        [string[]]$Arguments
    )
    try {
        $Output = & $FilePath @Arguments 2>$null
        if ($LASTEXITCODE -eq 0) {
            return (($Output | Out-String).Trim())
        }
    }
    catch {
    }
    return ""
}

function Write-BuildInfo {
    param(
        [string]$OutputPath
    )
    $Version = $env:MEMORY_EVAL_VERSION
    if ([string]::IsNullOrWhiteSpace($Version)) {
        $Version = "local"
    }
    $GitCommit = Invoke-Capture "git" @("-C", $Root, "rev-parse", "--short", "HEAD")
    $GitBranch = Invoke-Capture "git" @("-C", $Root, "rev-parse", "--abbrev-ref", "HEAD")
    $GitStatus = Invoke-Capture "git" @("-C", $Root, "status", "--porcelain")
    $PythonVersion = Invoke-Capture $Python @("--version")
    $BuildInfo = [ordered]@{
        app_name = "MemoryEvalUI"
        version = $Version
        built_at_utc = (Get-Date).ToUniversalTime().ToString("o")
        git_commit = $GitCommit
        git_branch = $GitBranch
        git_dirty = -not [string]::IsNullOrWhiteSpace($GitStatus)
        build_mode = "release"
        python = $Python
        python_version = $PythonVersion
    }
    $OutputDir = Split-Path -Parent $OutputPath
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    $BuildInfo | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $OutputPath -Encoding utf8
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
    Invoke-Checked $Python @("scripts\smoke_pages.py") "page smoke tests"
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
Write-BuildInfo (Join-Path $WorkspaceTemp "build_info.json")
Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue
Invoke-Checked $Python @("-m", "PyInstaller", "MemoryEvalUI.spec", "--clean", "--noconfirm") "PyInstaller"

$Exe = Join-Path $Root "dist\MemoryEvalUI\MemoryEvalUI.exe"
if (-not (Test-Path $Exe)) {
    throw "Build finished but exe was not found: $Exe"
}

$ReleaseRoot = Join-Path $Root "dist\MemoryEvalUI"
$WritableDirs = @(
    "data\cases"
    "data\raw\uploads"
    "data\results"
    "config"
    "logs"
    "prompts\judge"
    "prompts\generation"
)
foreach ($RelativeDir in $WritableDirs) {
    $TargetDir = Join-Path $ReleaseRoot $RelativeDir
    New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null
    New-Item -ItemType File -Force -Path (Join-Path $TargetDir ".gitkeep") | Out-Null
}
$ExampleConfig = Join-Path $Root "config\local_config.example.json"
if (Test-Path $ExampleConfig) {
    Copy-Item -LiteralPath $ExampleConfig -Destination (Join-Path $ReleaseRoot "config\local_config.example.json") -Force
}
$BuildInfoPath = Join-Path $WorkspaceTemp "build_info.json"
if (Test-Path $BuildInfoPath) {
    Copy-Item -LiteralPath $BuildInfoPath -Destination (Join-Path $ReleaseRoot "build_info.json") -Force
}
$PromptSource = Join-Path $Root "prompts"
$PromptTarget = Join-Path $ReleaseRoot "prompts"
if (Test-Path $PromptSource) {
    Get-ChildItem -LiteralPath (Join-Path $PromptSource "judge") -File -ErrorAction SilentlyContinue |
        Copy-Item -Destination (Join-Path $PromptTarget "judge") -Force
    Get-ChildItem -LiteralPath (Join-Path $PromptSource "generation") -File -ErrorAction SilentlyContinue |
        Copy-Item -Destination (Join-Path $PromptTarget "generation") -Force
}

if (-not $SkipSmoke) {
    $PreviousPort = $env:STREAMLIT_SERVER_PORT
    $PreviousAddress = $env:STREAMLIT_SERVER_ADDRESS
    $PreviousHeadless = $env:STREAMLIT_SERVER_HEADLESS
    $env:STREAMLIT_SERVER_PORT = "18501"
    $env:STREAMLIT_SERVER_ADDRESS = "127.0.0.1"
    $env:STREAMLIT_SERVER_HEADLESS = "true"
    $Proc = $null
    try {
        $Proc = Start-Process -FilePath $Exe -WindowStyle Hidden -PassThru
        $HealthUrl = "http://127.0.0.1:18501/_stcore/health"
        $Ready = $false
        $Deadline = (Get-Date).AddSeconds(45)
        while ((Get-Date) -lt $Deadline) {
            if ($Proc.HasExited) {
                throw "Smoke test failed: MemoryEvalUI.exe exited early with code $($Proc.ExitCode)"
            }
            try {
                $Response = Invoke-WebRequest -Uri $HealthUrl -UseBasicParsing -TimeoutSec 2
                if ($Response.StatusCode -eq 200) {
                    $Ready = $true
                    break
                }
            }
            catch {
                Start-Sleep -Milliseconds 500
            }
        }
        if (-not $Ready) {
            throw "Smoke test failed: health endpoint did not become ready within 45 seconds"
        }
        Write-Host "Executable smoke test passed: $HealthUrl"
    }
    finally {
        if ($null -ne $Proc -and -not $Proc.HasExited) {
            Stop-Process -Id $Proc.Id -Force
            $Proc.WaitForExit()
        }
        $env:STREAMLIT_SERVER_PORT = $PreviousPort
        $env:STREAMLIT_SERVER_ADDRESS = $PreviousAddress
        $env:STREAMLIT_SERVER_HEADLESS = $PreviousHeadless
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
