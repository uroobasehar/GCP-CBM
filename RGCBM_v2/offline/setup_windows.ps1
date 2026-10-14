param([switch]$SkipGpuCheck)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# Verify the complete distribution before executing bundled installers/packages.
foreach ($line in Get-Content (Join-Path $Root 'SHA256SUMS.txt')) {
    if ($line.Trim()) {
        $parts = $line -split '  ', 2
        $path = Join-Path $Root $parts[1]
        if (!(Test-Path -LiteralPath $path)) { throw "Missing bundle file: $($parts[1])" }
        $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLower()
        if ($actual -ne $parts[0]) { throw "Checksum mismatch: $($parts[1])" }
    }
}

$Redist = Join-Path $PSScriptRoot 'installers\vc_redist.x64.exe'
if (Test-Path $Redist) {
    $signature = Get-AuthenticodeSignature $Redist
    if ($signature.Status -ne 'Valid') { throw 'Microsoft VC++ installer signature is not valid. Resolve Windows certificate trust; do not disable verification.' }
    $process = Start-Process $Redist -ArgumentList '/install', '/quiet', '/norestart' -Wait -PassThru
    if ($process.ExitCode -notin @(0, 1638, 3010)) { throw "VC++ installer failed: $($process.ExitCode)" }
}

$Conda = Get-Command conda -ErrorAction SilentlyContinue
if (!$Conda) { throw 'Open an Anaconda PowerShell Prompt with conda initialized, then rerun this script.' }
$Environment = Join-Path $Root 'conda-env'
$env:CONDA_PKGS_DIRS = Join-Path $PSScriptRoot 'conda-cache'
$Explicit = Join-Path $PSScriptRoot 'conda_explicit_local.txt'
$urls = @('@EXPLICIT')
foreach ($line in Get-Content (Join-Path $PSScriptRoot 'conda_package_files.txt')) {
    if ($line.Trim()) {
        $absolute = (Resolve-Path (Join-Path $PSScriptRoot ('conda-packages\' + $line.Trim()))).Path
        $urls += ([System.Uri]$absolute).AbsoluteUri
    }
}
# Local file URLs avoid both a solver and all remote repodata/network requests.
[System.IO.File]::WriteAllLines($Explicit, $urls, [System.Text.UTF8Encoding]::new($false))
if (!(Test-Path (Join-Path $Environment 'python.exe'))) {
    & conda create --offline --yes --prefix $Environment --file $Explicit
    if ($LASTEXITCODE -ne 0) { throw 'Offline conda environment creation failed.' }
}
$Python = Join-Path $Environment 'python.exe'
$env:PATH = "$Environment;$Environment\Library\bin;$Environment\Scripts;$env:PATH"
$version = & $Python -c "import sys; print('%d.%d' % sys.version_info[:2])"
if ($version -ne '3.11') { throw 'This wheelhouse requires 64-bit CPython 3.11.' }
$VenvPython = $Python
& $VenvPython -m pip install --no-index --find-links (Join-Path $PSScriptRoot 'wheelhouse') --require-hashes -r (Join-Path $PSScriptRoot 'lock-win-py311.txt')
if ($LASTEXITCODE -ne 0) { throw 'Offline package installation failed.' }
& $VenvPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Installed package requirements are inconsistent.' }
& $VenvPython -m ipykernel install --prefix $Environment --name rgcbm-offline --display-name 'RG-CBM Anaconda offline GPU (Python 3.11)'
if ($LASTEXITCODE -ne 0) { throw 'Jupyter kernel registration failed.' }
if (!$SkipGpuCheck) {
    & $VenvPython (Join-Path $PSScriptRoot 'gpu_check.py') --require-gpu
    if ($LASTEXITCODE -ne 0) { throw 'GPU validation failed; check the NVIDIA driver before training.' }
}
Write-Host 'Offline dependencies are installed. Run offline\launch_jupyter.cmd and open notebooks\RGCBM_Offline.ipynb.'
