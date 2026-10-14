param([string]$Output = (Join-Path $PSScriptRoot 'RGCBM_Offline.zip'))
$ErrorActionPreference = 'Stop'
$manifest = Get-Content (Join-Path $PSScriptRoot 'bundle_manifest.json') -Raw | ConvertFrom-Json
if (Test-Path -LiteralPath $Output) { throw 'Output already exists. Verify/preserve it before choosing a new output path.' }
foreach ($part in $manifest.parts) {
    $path = Join-Path $PSScriptRoot $part.name
    if (!(Test-Path -LiteralPath $path)) { throw "Missing archive part: $($part.name). Run git lfs pull before going offline." }
    if ((Get-Item -LiteralPath $path).Length -ne $part.bytes) { throw "Wrong part size: $($part.name). A Git LFS pointer is not the archive content; run git lfs pull." }
    $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLower()
    if ($actual -ne $part.sha256) { throw "Archive part checksum mismatch: $($part.name)" }
}
$target = [System.IO.File]::Open($Output, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write)
try {
    foreach ($part in $manifest.parts) {
        $source = [System.IO.File]::OpenRead((Join-Path $PSScriptRoot $part.name))
        try { $source.CopyTo($target, 1048576) } finally { $source.Dispose() }
    }
} finally { $target.Dispose() }
if ((Get-FileHash -LiteralPath $Output -Algorithm SHA256).Hash.ToLower() -ne $manifest.archive_sha256) {
    throw 'Joined ZIP checksum mismatch; do not install it.'
}
Write-Host "Verified offline ZIP created: $Output"
Write-Host 'Extract it, open an Anaconda PowerShell Prompt in RGCBM_MultiDataset, and run offline\setup_windows.ps1.'
