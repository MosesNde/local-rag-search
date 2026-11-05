
# LUCY - Windows source installation and launcher
# Run: powershell -ExecutionPolicy Bypass -File .\setup-windows.ps1

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

function Log([string]$Message) {
    Write-Host "`n[LUCY] $Message"
}

function Check-Exit([string]$Task) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Task failed (exit code $LASTEXITCODE)."
    }
}

function Find-Python314 {
    $paths = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python314\python.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python314-arm64\python.exe'),
        (Join-Path $env:ProgramFiles 'Python314\python.exe')
    )

    foreach ($path in $paths) {
        if (Test-Path $path) {
            return $path
        }
    }

    # Check the Python launcher without failing on missing versions.
    if (Get-Command py -ErrorAction SilentlyContinue) {
        try {
            $result = & py -3.14 -c 'import sys; print(sys.executable)' 2>$null

            if ($LASTEXITCODE -eq 0 -and $result) {
                return $result.Trim()
            }
        }
        catch {
            # Continue to automatic installation.
        }
    }

    return $null
}

function Find-Ollama {
    $command = Get-Command ollama -ErrorAction SilentlyContinue

    if ($command) {
        return $command.Source
    }

    $paths = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe'),
        (Join-Path $env:LOCALAPPDATA 'Ollama\ollama.exe'),
        (Join-Path $env:ProgramFiles 'Ollama\ollama.exe')
    )

    foreach ($path in $paths) {
        if (Test-Path $path) {
            return $path
        }
    }

    return $null
}

function Test-OllamaServer {
    try {
        $null = Invoke-WebRequest `
            -Uri 'http://127.0.0.1:11434/api/tags' `
            -TimeoutSec 3 `
            -UseBasicParsing

        return $true
    }
    catch {
        return $false
    }
}

# --------------------------------------------------
# 1. Python
# --------------------------------------------------

Log 'Checking Python 3.14'

$python = Find-Python314

if (-not $python) {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw 'Python 3.14 is missing and winget is unavailable.'
    }

    Log 'Installing Python 3.14'

    & winget install `
        --id Python.Python.3.14 `
        --exact `
        --source winget `
        --accept-source-agreements `
        --accept-package-agreements

    Check-Exit 'Python installation'

    $python = Find-Python314

    if (-not $python) {
        throw 'Python 3.14 was installed but could not be located. Restart PowerShell and rerun.'
    }
}

Log "Using Python: $python"

# --------------------------------------------------
# 2. Virtual environment
# --------------------------------------------------

$venv = Join-Path $PSScriptRoot '.venv-lucy'
$venvPython = Join-Path $venv 'Scripts\python.exe'

# Preserve existing environments using another version.
if (Test-Path $venvPython) {
    $version = & $venvPython -c 'import sys; print(sys.version_info.major, sys.version_info.minor)'
    Check-Exit 'Python version check'

    if ($version.Trim() -ne '3 14') {
        Log 'Existing .venv-lucy uses another Python version; preserving it.'

        $venv = Join-Path $PSScriptRoot '.venv-lucy-314'
        $venvPython = Join-Path $venv 'Scripts\python.exe'
    }
}

if (Test-Path $venvPython) {
    $version = & $venvPython -c 'import sys; print(sys.version_info.major, sys.version_info.minor)'
    Check-Exit 'Python environment check'

    if ($version.Trim() -ne '3 14') {
        throw 'The selected LUCY environment is not Python 3.14.'
    }
}
else {
    Log 'Creating isolated Python environment'

    & $python -m venv $venv

    Check-Exit 'Virtual environment creation'
}

# --------------------------------------------------
# 3. Python dependencies
# --------------------------------------------------

Log 'Installing Python dependencies'

& $venvPython -m pip install `
    --index-url https://pypi.org/simple `
    --timeout 30 `
    --retries 2 `
    -r (Join-Path $PSScriptRoot 'requirements.txt')

Check-Exit 'Python dependencies installation'

& $venvPython -m pip check
Check-Exit 'Dependency validation'

# --------------------------------------------------
# 4. Ollama
# --------------------------------------------------

Log 'Checking Ollama'

$ollama = Find-Ollama

if (-not $ollama) {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw 'Ollama is missing and winget is unavailable.'
    }

    Log 'Installing Ollama'

    & winget install `
        --id Ollama.Ollama `
        --exact `
        --source winget `
        --accept-source-agreements `
        --accept-package-agreements

    Check-Exit 'Ollama installation'

    $ollama = Find-Ollama

    if (-not $ollama) {
        throw 'Ollama was installed but its executable was not found.'
    }
}

# --------------------------------------------------
# 5. Start Ollama
# --------------------------------------------------

if (-not (Test-OllamaServer)) {
    Log 'Starting Ollama server'

    Start-Process -FilePath $ollama `
        -ArgumentList 'serve' `
        -WindowStyle Hidden | Out-Null
}

$ready = $false

for ($i = 0; $i -lt 45; $i++) {
    if (Test-OllamaServer) {
        $ready = $true
        break
    }

    Start-Sleep -Seconds 2
}

if (-not $ready) {
    throw 'Ollama did not start. Launch Ollama manually and rerun setup.'
}

# --------------------------------------------------
# 6. AI models
# --------------------------------------------------

Log 'Checking AI models'

$installedModels = @(
    & $ollama list | Select-Object -Skip 1 |
        ForEach-Object { ($_ -split '\s+')[0] }
)

Check-Exit 'Ollama model listing'

foreach ($model in @(
    'nomic-embed-text',
    'qwen3:8b',
    'qwen3-vl:8b'
)) {
    $normalizedModel = $model

    if ($model -notmatch ':') {
        $normalizedModel = "${model}:latest"
    }

    if ($normalizedModel -notin $installedModels) {
        Log "Downloading $model"

        & $ollama pull $model
        Check-Exit "Model download: $model"
    }
    else {
        Log "$model already installed"
    }
}

# --------------------------------------------------
# 7. Launch LUCY
# --------------------------------------------------

Log 'Starting LUCY at http://localhost:8501'

& $venvPython -m streamlit run `
    (Join-Path $PSScriptRoot 'src\web_app.py')

Check-Exit 'LUCY startup'
