$projectRoot = $PSScriptRoot
$venvEntry = Join-Path $projectRoot ".venv\Scripts\rag-module-test.exe"
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"
$exitCode = 127

Push-Location -LiteralPath $projectRoot
try {
    if (Test-Path -LiteralPath $venvEntry -PathType Leaf) {
        & $venvEntry @args
        $exitCode = $LASTEXITCODE
    }
    elseif (Test-Path -LiteralPath $venvPython -PathType Leaf) {
        & $venvPython -m rag_system.module_test @args
        $exitCode = $LASTEXITCODE
    }
    else {
        $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if ($null -eq $pythonCommand) {
            Write-Error "Python was not found. Create .venv or install Python 3.10+."
        }
        else {
            & $pythonCommand.Source -m rag_system.module_test @args
            $exitCode = $LASTEXITCODE
        }
    }
}
finally {
    Pop-Location
}

exit $exitCode
