$ErrorActionPreference = 'Stop'

$psql = 'C:\Program Files\PostgreSQL\18\bin\psql.exe'
$envFile = Join-Path (Split-Path -Parent $PSScriptRoot) '.env'
if (-not (Test-Path -LiteralPath $psql)) {
    throw 'PostgreSQL 18 psql.exe was not found.'
}

$credential = Get-Credential -UserName 'postgres' -Message 'Enter the local PostgreSQL administrator password'
$pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($credential.Password)
try {
    $env:PGPASSWORD = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    $appPassword = ([Guid]::NewGuid().ToString('N') + [Guid]::NewGuid().ToString('N'))
    $roleExists = & $psql -w -h 127.0.0.1 -U postgres -d postgres -tAc "SELECT 1 FROM pg_roles WHERE rolname='cognitrack'"
    if ($LASTEXITCODE -ne 0) { throw 'PostgreSQL administrator authentication failed.' }
    if ([string]$roleExists -eq '1') {
        & $psql -w -h 127.0.0.1 -U postgres -d postgres -v ON_ERROR_STOP=1 -c "ALTER ROLE cognitrack WITH LOGIN PASSWORD '$appPassword'"
    } else {
        & $psql -w -h 127.0.0.1 -U postgres -d postgres -v ON_ERROR_STOP=1 -c "CREATE ROLE cognitrack WITH LOGIN PASSWORD '$appPassword'"
    }
    $databaseExists = & $psql -w -h 127.0.0.1 -U postgres -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='cognitrack'"
    if ([string]$databaseExists -ne '1') {
        & $psql -w -h 127.0.0.1 -U postgres -d postgres -v ON_ERROR_STOP=1 -c 'CREATE DATABASE cognitrack OWNER cognitrack'
    }
    $databaseUrl = "DATABASE_URL=postgresql://cognitrack:$appPassword@127.0.0.1:5432/cognitrack"
    $lines = if (Test-Path -LiteralPath $envFile) { Get-Content -LiteralPath $envFile } else { @() }
    $lines = @($lines | Where-Object { $_ -notmatch '^DATABASE_URL=' }) + $databaseUrl
    [IO.File]::WriteAllLines($envFile, $lines, [Text.UTF8Encoding]::new($false))
    Write-Host 'PostgreSQL is configured. Restart the CogniTrack server to activate it.' -ForegroundColor Green
} finally {
    Remove-Item Env:PGPASSWORD -ErrorAction SilentlyContinue
    if ($pointer -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    }
}
