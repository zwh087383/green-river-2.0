param(
    [int]$ScreenIndex = -1
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms

$screens = @([System.Windows.Forms.Screen]::AllScreens)
if ($ScreenIndex -ge 0) {
    if ($ScreenIndex -ge $screens.Count) {
        throw "屏幕编号 $ScreenIndex 不存在。当前检测到 $($screens.Count) 块屏幕。"
    }
    $screen = $screens[$ScreenIndex]
} else {
    $screen = $screens | Where-Object { -not $_.Primary } | Select-Object -First 1
}

if (-not $screen) {
    Write-Host '当前只检测到电脑主屏幕；未打开表情窗口。外接屏连接后，请使用 Windows「扩展」模式并重新运行启动脚本。'
    exit 0
}

$browserCandidates = @(
    'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
    'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
    'C:\Program Files\Google\Chrome\Application\chrome.exe',
    'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe'
)
$browserPath = $browserCandidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
if (-not $browserPath) {
    throw '没有找到 Microsoft Edge 或 Google Chrome，无法在外接屏打开表情页。'
}

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$profilePath = Join-Path $projectRoot 'data\runtime\display-browser-profile'
New-Item -ItemType Directory -Path $profilePath -Force | Out-Null
$bounds = $screen.Bounds
$displayUrl = 'http://127.0.0.1:8000/display'

function Get-DisplayWindow {
    $candidates = Get-CimInstance Win32_Process -Filter "Name = 'msedge.exe' OR Name = 'chrome.exe'" |
        Where-Object { $_.CommandLine -like "*--app=$displayUrl*" -and $_.CommandLine -like "*$profilePath*" }
    foreach ($candidate in $candidates) {
        $process = Get-Process -Id $candidate.ProcessId -ErrorAction SilentlyContinue
        if ($process -and $process.MainWindowHandle -ne 0) { return $process }
    }
    return $null
}

function Set-DisplayTopmost($process) {
    if (-not ('GreenRiverDisplayWindow' -as [type])) {
        Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class GreenRiverDisplayWindow {
    [DllImport("user32.dll")]
    public static extern bool SetWindowPos(IntPtr window, IntPtr after, int x, int y, int width, int height, uint flags);
}
'@
    }
    # Keep other windows' borders and shadows from covering the HDMI screen.
    $noMoveNoResizeNoActivate = [uint32]0x13
    if (-not [GreenRiverDisplayWindow]::SetWindowPos(
        $process.MainWindowHandle, [IntPtr](-1), 0, 0, 0, 0, $noMoveNoResizeNoActivate
    )) {
        Write-Warning '表情窗口已打开，但未能置于最前；请检查外接屏画面。'
    }
}

$existingWindow = Get-DisplayWindow
if ($existingWindow) {
    Set-DisplayTopmost $existingWindow
    Write-Host "表情窗口已经打开：进程 $($existingWindow.Id)。"
    exit 0
}

$browserArgs = @(
    "--user-data-dir=`"$profilePath`"",
    '--no-first-run',
    '--no-default-browser-check',
    "--app=$displayUrl",
    "--window-position=$($bounds.X),$($bounds.Y)",
    "--window-size=$($bounds.Width),$($bounds.Height)",
    '--start-fullscreen'
)

Start-Process -FilePath $browserPath -ArgumentList $browserArgs
$displayWindow = $null
for ($attempt = 0; $attempt -lt 40; $attempt++) {
    $displayWindow = Get-DisplayWindow
    if ($displayWindow) { break }
    Start-Sleep -Milliseconds 250
}
if ($displayWindow) {
    Set-DisplayTopmost $displayWindow
} else {
    Write-Warning '浏览器进程已启动，但尚未找到表情窗口；请检查外接屏画面。'
}
Write-Host "表情页已在外接屏打开：$($screen.DeviceName)（$($bounds.Width)×$($bounds.Height)）。"
