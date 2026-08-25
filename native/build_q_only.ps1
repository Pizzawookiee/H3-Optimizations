# Build the benchmark-only Q quantizer side-car for one GPU architecture.
# This does not replace or modify native/bin/h3_int8_attention.dll.

[CmdletBinding()]
param(
    [string]$Arch = "",
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$nvcc = (Get-Command nvcc -ErrorAction SilentlyContinue).Source
if (-not $nvcc) {
    $toolkit = Join-Path ${env:ProgramFiles} "NVIDIA GPU Computing Toolkit\CUDA"
    $found = Get-ChildItem $toolkit -Filter nvcc.exe -Recurse -ErrorAction SilentlyContinue |
        Sort-Object FullName -Descending | Select-Object -First 1
    if (-not $found) { throw "nvcc was not found; install the CUDA toolkit or put nvcc on PATH" }
    $nvcc = $found.FullName
}
if (-not $Arch) {
    $query = & nvidia-smi --query-gpu=compute_cap --format=csv,noheader
    if ($LASTEXITCODE -ne 0 -or -not $query) { throw "could not query compute capability; pass -Arch" }
    $Arch = ($query -split "`n")[0].Trim().Replace(".", "")
}

$relative = Join-Path "VC" (Join-Path "Auxiliary" (Join-Path "Build" "vcvars64.bat"))
$vcvars = ""
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vswhere) {
    $root = & $vswhere -latest -products * -property installationPath
    if ($root) { $vcvars = Join-Path $root $relative }
}
if (-not $vcvars -or -not (Test-Path $vcvars)) {
    $studios = Join-Path ${env:ProgramFiles} "Microsoft Visual Studio"
    $vcvars = (Get-ChildItem $studios -Filter vcvars64.bat -Recurse -ErrorAction SilentlyContinue |
        Select-Object -First 1).FullName
}
if (-not $vcvars -or -not (Test-Path $vcvars)) {
    throw "vcvars64.bat was not found; install the Visual Studio C++ build tools"
}

if (-not $Output) { $Output = Join-Path $here "bin\h3_q_only.dll" }
$outputDir = Split-Path -Parent $Output
if (-not (Test-Path $outputDir)) { New-Item -ItemType Directory -Force $outputDir | Out-Null }
$sources = @(
    (Join-Path $here "src\h3_q_only_api.cu"),
    (Join-Path $here "src\sage_attention\quant_qk_int8.cu")
)
$arguments = @(
    "-shared", "-O3", "--use_fast_math", "--expt-relaxed-constexpr",
    "--expt-extended-lambda", "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-std=c++20",
    "-gencode", "arch=compute_$Arch,code=sm_$Arch",
    "-I", (Join-Path $here "src"),
    "-I", (Join-Path $here "src\sage_attention"),
    "-o", $Output
) + $sources
$quoted = ($arguments | ForEach-Object { '"' + $_ + '"' }) -join " "
$line = 'call "' + $vcvars + '" >nul && "' + $nvcc + '" ' + $quoted
& cmd.exe /c $line
if ($LASTEXITCODE -ne 0) { throw "nvcc failed with exit code $LASTEXITCODE" }
if (-not (Test-Path $Output)) { throw "nvcc reported success but $Output is missing" }
Write-Host "built $Output for sm_$Arch"
