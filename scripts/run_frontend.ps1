param(
    [switch]$Build
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$frontendRoot = Join-Path $projectRoot "frontend"
$bundledNode = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe"
$vite = Join-Path $frontendRoot "node_modules\vite\bin\vite.js"

if (-not (Test-Path -LiteralPath $vite)) {
    throw "Frontend dependencies are missing. Install them from the frontend folder before starting Vite."
}

if (Test-Path -LiteralPath $bundledNode) {
    $nodeExecutable = $bundledNode
} else {
    $nodeExecutable = (Get-Command node -ErrorAction Stop).Source
    $majorVersion = [int]((& $nodeExecutable --version).TrimStart("v").Split(".")[0])
    if ($majorVersion -lt 20) {
        throw "CineMatch requires Node.js 20 or newer. The detected version is too old."
    }
}

Push-Location $frontendRoot
try {
    if ($Build) {
        & $nodeExecutable $vite build
    } else {
        & $nodeExecutable $vite --host 127.0.0.1 --port 5173
    }
} finally {
    Pop-Location
}
