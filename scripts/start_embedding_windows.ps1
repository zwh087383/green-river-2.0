param([string]$ModelDir = $env:EMBEDDINGGEMMA_MODEL_DIR, [string]$EmbeddingUrl)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if (-not $EmbeddingUrl) {
    $dataRoot = if ($env:RAG_DATA_DIR) { $env:RAG_DATA_DIR } else { Join-Path $projectRoot 'data' }
    $settingsPath = Join-Path $dataRoot 'settings.json'
    if (Test-Path -LiteralPath $settingsPath) {
        $savedSettings = Get-Content -LiteralPath $settingsPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $EmbeddingUrl = $savedSettings.embedding_url
    }
}
if ($EmbeddingUrl -and $EmbeddingUrl.TrimEnd('/') -notin @('http://127.0.0.1:8765/v1/embeddings', 'http://localhost:8765/v1/embeddings')) {
    Write-Host 'Using the configured embedding endpoint; local model auto-start skipped.'
    return
}
$healthUrl = 'http://127.0.0.1:8765/health'
function Test-EmbeddingService {
    try {
        $health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2
        return ($health.status -eq 'ready' -and $health.model -eq 'google/embeddinggemma-2' -and $health.dimensions -eq 768)
    } catch { return $false }
}
if (Test-EmbeddingService) {
    Write-Host 'EmbeddingGemma 2 is ready at http://127.0.0.1:8765'
    return
}
if (Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue) {
    throw 'Port 8765 is occupied but the expected EmbeddingGemma 2 service is not ready.'
}
if (-not $ModelDir -and (Test-Path -LiteralPath 'F:\AI-Shared\model_test\embeddinggemma2\serve.py')) {
    $ModelDir = 'F:\AI-Shared\model_test\embeddinggemma2'
}
if (-not $ModelDir) {
    throw 'Set EMBEDDINGGEMMA_MODEL_DIR or pass -ModelDir pointing to the existing model service directory.'
}
$modelRoot = (Resolve-Path -LiteralPath $ModelDir).Path
$pythonPath = Join-Path $modelRoot '.venv\Scripts\python.exe'
$serverPath = Join-Path $modelRoot 'serve.py'
if (-not (Test-Path -LiteralPath $pythonPath) -or -not (Test-Path -LiteralPath $serverPath)) {
    throw 'The model directory must contain serve.py and its .venv Python runtime.'
}
$runtimeDir = Join-Path $projectRoot 'data\runtime'
New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
$modelProcess = Start-Process -FilePath $pythonPath -ArgumentList @('-u', ('"' + $serverPath + '"')) `
    -WorkingDirectory $modelRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $runtimeDir 'embeddinggemma2.stdout.log') `
    -RedirectStandardError (Join-Path $runtimeDir 'embeddinggemma2.stderr.log')
$modelProcess.Id | Set-Content -LiteralPath (Join-Path $runtimeDir 'embeddinggemma2.pid')
for ($attempt = 0; $attempt -lt 120; $attempt++) {
    if ($modelProcess.HasExited) { throw 'Embedding service exited; inspect data/runtime/embeddinggemma2.stderr.log.' }
    if (Test-EmbeddingService) { Write-Host 'EmbeddingGemma 2 is ready.'; return }
    Start-Sleep -Milliseconds 500
}
throw 'Embedding service is still loading; inspect data/runtime/embeddinggemma2.stderr.log.'
