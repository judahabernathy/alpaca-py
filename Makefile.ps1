param(
    [Parameter(Position = 0)]
    [string]
    $Task = "help",

    [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
    [string[]]
    $TaskArgs
)

function Invoke-EdgeOpenApi {
    Write-Host "Exporting OpenAPI specification..." -ForegroundColor Cyan
    poetry run python tools/export_openapi.py
}

function Invoke-EdgeTest {
    $argsToPass = @()
    if ($TaskArgs) { $argsToPass = $TaskArgs }
    Write-Host "Running pytest $($argsToPass -join ' ')" -ForegroundColor Cyan
    poetry run pytest @argsToPass
}

function Invoke-EdgeServe {
    $port = if ($TaskArgs -and $TaskArgs.Length -gt 0) {
        $TaskArgs[0]
    } elseif ($env:PORT) {
        $env:PORT
    } else {
        8000
    }
    Write-Host "Starting uvicorn on port $port..." -ForegroundColor Cyan
    poetry run uvicorn edge.app:app --reload --port $port
}

switch ($Task.ToLowerInvariant()) {
    "openapi" { Invoke-EdgeOpenApi }
    "test"    { Invoke-EdgeTest }
    "serve"   { Invoke-EdgeServe }
    "help" {
        Write-Host "Available tasks:" -ForegroundColor Cyan
        Write-Host "  openapi   Export ./.well-known OpenAPI files"
        Write-Host "  test [..] Run pytest (additional args forwarded)"
        Write-Host "  serve [port]  Launch uvicorn with optional port (default 8000 or \$env:PORT)"
    }
    default {
        Write-Error "Unknown task '$Task'. Use: openapi, test, serve, help."
        exit 1
    }
}
