param([Parameter(Mandatory=$true)][string]$Installer)
$ErrorActionPreference = 'Stop'
# Run on an ONLINE preparation computer after downloading the complete standalone
# driver from NVIDIA's official RTX Enterprise/RTX PRO Windows driver selector.
$source = (Resolve-Path -LiteralPath $Installer).Path
$signature = Get-AuthenticodeSignature $source
if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'NVIDIA') {
    throw 'A valid NVIDIA-signed complete Windows driver installer is required. Do not use NVIDIA App, a web installer, or an unofficial mirror.'
}
$destination = Join-Path $PSScriptRoot 'installers\nvidia_driver.exe'
if (Test-Path $destination) { throw 'Driver destination already exists; preserve/verify it before replacing it.' }
Copy-Item -LiteralPath $source -Destination $destination
$record = @{ file = 'installers/nvidia_driver.exe'; sha256 = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLower(); signer = $signature.SignerCertificate.Subject; bytes = (Get-Item $destination).Length; instruction = 'Install this verified standalone driver on the target Windows machine, reboot, then run preflight_windows.ps1 and GPU checks. This companion is collected separately from the immutable distribution ZIP.' }
$record | ConvertTo-Json | Set-Content (Join-Path $PSScriptRoot 'driver_manifest.json') -Encoding UTF8
Write-Host 'NVIDIA-signed driver copied. Transfer both the driver and driver_manifest.json alongside the complete offline distribution. Verify the SHA-256 before running it.'
