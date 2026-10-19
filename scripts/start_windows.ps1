param(
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$runtimeDir = Join-Path $projectRoot 'data\runtime'
$pidPath = Join-Path $runtimeDir 'web-windows.pid'
$healthUrl = 'http://127.0.0.1:8000/api/health'
$adminUrl = 'http://127.0.0.1:8000/'
$displayUrl = 'http://127.0.0.1:8000/display'
$displayLauncher = Join-Path $PSScriptRoot 'open_display_windows.ps1'

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Python 虚拟环境不存在：$pythonPath。请先运行 .\.venv\Scripts\python.exe -m pip install -e '.[web,indexing]'."
}

function Test-ProjectServer {
    try {
        $health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2
        return ($health.ok -eq $true -and $health.service -eq 'rag-system')
    } catch {
        return $false
    }
}

if (Test-ProjectServer) {
    & $pythonPath -m rag_system.display_state mode emotion | Out-Null
    Write-Host "项目已经运行。管理网页：$adminUrl"
    if (-not $NoBrowser) {
        Start-Process $adminUrl
        & $displayLauncher
    }
    exit 0
}

$portOwner = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($portOwner) {
    throw '端口 8000 已被其他程序占用，请先关闭该程序或设置 RAG_SYSTEM_PORT 使用其他端口。'
}

New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
$stdoutPath = Join-Path $runtimeDir 'web-windows.log'
$stderrPath = Join-Path $runtimeDir 'web-windows.error.log'
$serverProcess = Start-Process -FilePath $pythonPath `
    -ArgumentList '-m', 'rag_system.web' `
    -WorkingDirectory $projectRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -PassThru
Set-Content -LiteralPath $pidPath -Value $serverProcess.Id -Encoding ascii

for ($attempt = 0; $attempt -lt 40; $attempt++) {
    if (Test-ProjectServer) {
        & $pythonPath -m rag_system.display_state mode emotion | Out-Null
        & $pythonPath -m rag_system.display_state emotion idle | Out-Null
        Write-Host "项目已启动。管理网页：$adminUrl"
        Write-Host "表情页：$displayUrl"
        Write-Host "进程 ID：$($serverProcess.Id)"
        if (-not $NoBrowser) {
            Start-Process $adminUrl
            & $displayLauncher
        }
        exit 0
    }
    if ($serverProcess.HasExited) { break }
    Start-Sleep -Milliseconds 500
}

Write-Error "启动失败。请查看 $stderrPath"
