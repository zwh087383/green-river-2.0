$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$pidPath = Join-Path $projectRoot 'data\runtime\web-windows.pid'
$expectedPython = (Join-Path $projectRoot '.venv\Scripts\python.exe')

if (-not (Test-Path -LiteralPath $pidPath -PathType Leaf)) {
    Write-Host '没有由 start_windows.ps1 记录的服务进程。'
    exit 0
}

$serverPid = [int](Get-Content -LiteralPath $pidPath -Raw).Trim()
$processInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $serverPid"
if ($processInfo) {
    $samePython = [string]::Equals($processInfo.ExecutablePath, $expectedPython, [StringComparison]::OrdinalIgnoreCase)
    $rightCommand = $processInfo.CommandLine -match 'rag_system\.web'
    if (-not ($samePython -and $rightCommand)) {
        throw "PID $serverPid 已被其他进程占用；没有停止该进程。"
    }
    # Windows venv launchers can spawn the base interpreter as a child process.
    $children = Get-CimInstance Win32_Process -Filter "ParentProcessId = $serverPid"
    foreach ($child in $children) {
        if ($child.CommandLine -match 'rag_system\.web' -and
            [IO.Path]::GetFileName($child.ExecutablePath) -ieq 'python.exe') {
            Stop-Process -Id $child.ProcessId -ErrorAction SilentlyContinue
        }
    }
    Stop-Process -Id $serverPid
}
Remove-Item -LiteralPath $pidPath
Write-Host '项目 Web 服务已停止。'
