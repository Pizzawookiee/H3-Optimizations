# Build the side-car V staging library with nvcc directly.
#
# The main library needs CMake and produces a multi-architecture 33 MB binary
# whose per-GPU self-test gates the sparse backend. This target is two kernels
# depending on nothing beyond the CUDA toolkit, so it builds for the local
# architecture in seconds and cannot affect the shipped binary. Once the main
# library is rebuilt through CMake it exports the same symbols and the loader
# prefers it; delete this .dll then.
#
#   .\native\build_v_staging.ps1                 # local GPU architecture
#   .\native\build_v_staging.ps1 -Arch 89        # explicit

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
Write-Host "nvcc: $nvcc"

if (-not $Arch) {
    $query = & nvidia-smi --query-gpu=compute_cap --format=csv,noheader
    if ($LASTEXITCODE -ne 0 -or -not $query) { throw "could not query the GPU compute capability; pass -Arch" }
    $Arch = ($query -split "`n")[0].Trim().Replace(".", "")
}
Write-Host "architecture: sm_$Arch"

# nvcc needs MSVC's cl.exe plus its INCLUDE/LIB environment on Windows, and
# -ccbin supplies only the first of those. Run the compile inside the VC
# developer environment instead.
$vcvarsRelative = Join-Path "VC" (Join-Path "Auxiliary" (Join-Path "Build" "vcvars64.bat"))
$vcvars = ""
$vswhere = Join-Path ${env:ProgramFiles(x86)} (Join-Path "Microsoft Visual Studio" (Join-Path "Installer" "vswhere.exe"))
if (Test-Path $vswhere) {
    $vsRoot = & $vswhere -latest -products * -property installationPath
    if ($vsRoot) { $vcvars = Join-Path $vsRoot $vcvarsRelative }
}
if (-not $vcvars -or -not (Test-Path $vcvars)) {
    $studios = Join-Path ${env:ProgramFiles} "Microsoft Visual Studio"
    $vcvars = (Get-ChildItem $studios -Filter "vcvars64.bat" -Recurse -ErrorAction SilentlyContinue |
        Select-Object -First 1).FullName
}
if (-not $vcvars -or -not (Test-Path $vcvars)) {
    throw "vcvars64.bat was not found; install the Visual Studio C++ build tools"
}
Write-Host "vcvars: $vcvars"

if (-not $Output) { $Output = Join-Path $here (Join-Path "bin" "h3_v_staging.dll") }
$binDir = Split-Path -Parent $Output
if (-not (Test-Path $binDir)) { New-Item -ItemType Directory -Force $binDir | Out-Null }

$sources = @(
    (Join-Path $here (Join-Path "src" "h3_v_staging_api.cu")),
    (Join-Path $here (Join-Path "src" (Join-Path "sage_attention" "v_staging.cu")))
)
foreach ($source in $sources) {
    if (-not (Test-Path $source)) { throw "missing source: $source" }
}

$arguments = @(
    "-shared", "-O3", "--use_fast_math",
    "--expt-relaxed-constexpr", "--expt-extended-lambda",
    "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-std=c++20",
    "-gencode", "arch=compute_$Arch,code=sm_$Arch",
    "-I", (Join-Path $here "src"),
    "-I", (Join-Path $here (Join-Path "src" "sage_attention")),
    "-o", $Output
) + $sources

Write-Host "building $Output"
$quoted = ($arguments | ForEach-Object { '"' + $_ + '"' }) -join " "
$line = 'call "' + $vcvars + '" >nul && "' + $nvcc + '" ' + $quoted
& cmd.exe /c $line
if ($LASTEXITCODE -ne 0) { throw "nvcc failed with exit code $LASTEXITCODE" }
if (-not (Test-Path $Output)) { throw "nvcc reported success but $Output is missing" }
Write-Host "built $Output"
