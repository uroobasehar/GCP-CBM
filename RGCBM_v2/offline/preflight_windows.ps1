param([switch]$RequireEnvironment)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$issues = [System.Collections.Generic.List[string]]::new()
$warnings = [System.Collections.Generic.List[string]]::new()
$os = Get-CimInstance Win32_OperatingSystem
$machine = Get-CimInstance Win32_ComputerSystem
$report = [ordered]@{ timestamp = (Get-Date).ToString('o'); os = $os.Caption; build = $os.BuildNumber; architecture = $env:PROCESSOR_ARCHITECTURE; system_ram_gb = [math]::Round($machine.TotalPhysicalMemory / 1GB, 1); root = $Root }
if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64') { $issues.Add('Use 64-bit Windows x64 and a 64-bit Anaconda PowerShell Prompt.') }
if ([int]$os.BuildNumber -lt 22000) { $warnings.Add('This delivery targets Windows 11; this OS build is older.') }
if (!(Get-Command conda -ErrorAction SilentlyContinue)) { $issues.Add('Open an initialized Anaconda PowerShell Prompt. Anaconda3 must already be installed.') }
if ($machine.TotalPhysicalMemory -lt 16GB) { $warnings.Add('At least 16 GB system RAM is recommended; 32 GB is preferable when using the CPU LLM. VRAM is separate.') }
if ($Root.Length -gt 80) { $warnings.Add('Use a short writable path such as D:\RGCBM to avoid Windows path-length problems.') }
$drive = [System.IO.DriveInfo]::new([System.IO.Path]::GetPathRoot($Root))
$report.free_disk_gb = [math]::Round($drive.AvailableFreeSpace / 1GB, 1)
$report.filesystem = $drive.DriveFormat
if ($drive.DriveFormat -eq 'FAT32') { $issues.Add('Use NTFS or exFAT: FAT32 cannot hold the joined ZIP, local model files or large checkpoints.') }
if ($drive.AvailableFreeSpace -lt 20GB) { $issues.Add('Less than 20 GB is free after extraction. Free space for environment, extracted datasets and checkpoints before installing.') }
if (!(Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    $issues.Add('nvidia-smi is missing. Install the standalone NVIDIA RTX PRO Blackwell Windows driver before permanent disconnection.')
} else {
    $gpu = & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits
    if ($LASTEXITCODE -ne 0) { $issues.Add('nvidia-smi failed; resolve the system driver first.') }
    $report.gpus = @($gpu)
    foreach ($line in $gpu) {
        $fields = $line -split ','
        if ([version]$fields[1].Trim() -lt [version]'570.65') { $issues.Add('The Windows driver is older than the CUDA 12.8 release requirement. Transfer a current RTX PRO Blackwell driver installer.') }
    }
}
$redist = Join-Path $PSScriptRoot 'installers\vc_redist.x64.exe'
if (!(Test-Path $redist)) { $issues.Add('The bundled Microsoft VC++ x64 installer is missing.') } else {
    $signature = Get-AuthenticodeSignature $redist
    $report.microsoft_installer_signature = [string]$signature.Status
    if ($signature.Status -ne 'Valid') { $issues.Add('Microsoft installer signature/trust cannot be verified. Resolve Windows certificate trust before disconnection; do not bypass it.') }
}
$browser = Get-Command msedge.exe,chrome.exe,firefox.exe -ErrorAction SilentlyContinue
$edge = Join-Path ${env:ProgramFiles(x86)} 'Microsoft\Edge\Application\msedge.exe'
if (!$browser -and !(Test-Path $edge)) { $warnings.Add('Confirm a local web browser is installed; Jupyter requires a browser but no browser download is bundled.') }
if ($RequireEnvironment) {
    $python = Join-Path $Root 'conda-env\python.exe'
    if (!(Test-Path $python)) { $issues.Add('Run offline\setup_windows.ps1 to create the environment first.') }
}
$report.issues = @($issues.ToArray()); $report.warnings = @($warnings.ToArray())
$report | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $PSScriptRoot 'preflight_report.json') -Encoding UTF8
$report | ConvertTo-Json -Depth 5 | Write-Host
if ($issues.Count) { throw 'Offline preflight found prerequisites to resolve. See offline\preflight_report.json.' }
Write-Host 'Prerequisite checks passed. Run setup_windows.ps1, then validate_runtime.py, with the network physically disconnected before accepting the machine.'
