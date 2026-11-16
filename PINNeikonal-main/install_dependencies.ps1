<#
.SYNOPSIS
Creates the Python 3.8 environment for the PINNeikonal notebooks.
.EXAMPLE
powershell -NoProfile -ExecutionPolicy Bypass -File .\install_dependencies.ps1
.EXAMPLE
.\install_dependencies.ps1 -PythonExe "C:\Python38\python.exe"
.EXAMPLE
.\install_dependencies.ps1 -CheckOnly
#>
[CmdletBinding()]
param(
    [string]$PythonExe,
    [switch]$CheckOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-PythonInfo {
    param(
        [string]$Executable,
        [string[]]$PrefixArguments = @()
    )

    # Python single quotes also survive Windows PowerShell 5.1 native argument handling.
    $probeCode = "import json, struct, sys; print(json.dumps({'major': sys.version_info.major, 'minor': sys.version_info.minor, 'bits': struct.calcsize('P') * 8, 'executable': sys.executable, 'prefix': sys.prefix}))"
    try {
        $probeOutput = @(& $Executable @PrefixArguments -c $probeCode 2>$null)
        if ($LASTEXITCODE -ne 0) {
            return $null
        }
        return (($probeOutput -join "`n") | ConvertFrom-Json)
    }
    catch {
        return $null
    }
}

function Assert-Python38 {
    param($Info, [string]$Description)

    if ($null -eq $Info) {
        throw "$Description could not be started."
    }
    if ($Info.major -ne 3 -or $Info.minor -ne 8 -or $Info.bits -ne 64) {
        throw ("{0} uses Python {1}.{2} ({3}-bit). These notebooks require Python 3.8 (64-bit)." -f $Description, $Info.major, $Info.minor, $Info.bits)
    }
}

function Invoke-PythonStep {
    param(
        [string]$Executable,
        [string[]]$PythonArguments,
        [string]$Description
    )

    Write-Host "`n$Description"
    & $Executable @PythonArguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed (exit code $LASTEXITCODE). See the command output above."
    }
}

try {
    $projectRoot = $PSScriptRoot
    $requirementsPath = Join-Path $projectRoot 'requirements-legacy.txt'
    $venvPath = Join-Path $projectRoot '.venv'
    $venvPython = Join-Path $venvPath 'Scripts\python.exe'
    $codesPath = Join-Path $projectRoot 'codes'

    if (-not (Test-Path -LiteralPath $requirementsPath -PathType Leaf)) {
        throw "Requirements file not found: $requirementsPath"
    }

    if (Test-Path -LiteralPath $venvPath) {
        if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
            throw "The existing .venv is incomplete. Rename it before rerunning this installer."
        }
        $pythonInfo = Get-PythonInfo -Executable $venvPython
        Assert-Python38 -Info $pythonInfo -Description 'The existing .venv'
        if ([IO.Path]::GetFullPath($pythonInfo.prefix).TrimEnd('\') -ine [IO.Path]::GetFullPath($venvPath).TrimEnd('\')) {
            throw "The .venv interpreter does not belong to $venvPath. Rename that folder before rerunning."
        }
        Write-Host "Using existing environment: $venvPath"
    }
    else {
        $pythonInfo = $null
        if ($PythonExe) {
            $explicitCommand = Get-Command -Name $PythonExe -CommandType Application -ErrorAction SilentlyContinue
            if ($null -eq $explicitCommand) {
                throw "Python executable not found: $PythonExe"
            }
            $pythonInfo = Get-PythonInfo -Executable $explicitCommand.Source
            Assert-Python38 -Info $pythonInfo -Description $PythonExe
        }
        else {
            $launcher = Get-Command py.exe -CommandType Application -ErrorAction SilentlyContinue
            if ($null -ne $launcher) {
                $pythonInfo = Get-PythonInfo -Executable $launcher.Source -PrefixArguments @('-3.8')
            }
            if ($null -eq $pythonInfo) {
                $pythonCommand = Get-Command python.exe -CommandType Application -ErrorAction SilentlyContinue
                if ($null -ne $pythonCommand) {
                    $candidateInfo = Get-PythonInfo -Executable $pythonCommand.Source
                    if ($null -ne $candidateInfo -and $candidateInfo.major -eq 3 -and $candidateInfo.minor -eq 8) {
                        $pythonInfo = $candidateInfo
                    }
                }
            }
            if ($null -eq $pythonInfo) {
                throw ('Python 3.8 was not found. Install the Windows 64-bit Python 3.8.10 release from https://www.python.org/downloads/release/python-3810/ and rerun this script. If Python 3.8 is installed elsewhere, pass -PythonExe with its full python.exe path.')
            }
            Assert-Python38 -Info $pythonInfo -Description 'The selected interpreter'
        }
        Write-Host ("Using Python 3.8: {0}" -f $pythonInfo.executable)
    }

    if ($CheckOnly) {
        Write-Host 'Prerequisites passed. Run without -CheckOnly to install the packages and register the notebook kernel.'
        exit 0
    }

    if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
        Invoke-PythonStep -Executable $pythonInfo.executable -PythonArguments @('-m', 'venv', $venvPath) -Description 'Creating the project virtual environment'
        $createdInfo = Get-PythonInfo -Executable $venvPython
        Assert-Python38 -Info $createdInfo -Description 'The new .venv'
    }

    Invoke-PythonStep -Executable $venvPython -PythonArguments @('-m', 'pip', '--require-virtualenv', '--disable-pip-version-check', 'install', '--upgrade', 'pip<25.1', 'setuptools<81', 'wheel') -Description 'Installing Python 3.8-compatible packaging tools'
    Invoke-PythonStep -Executable $venvPython -PythonArguments @('-m', 'pip', '--require-virtualenv', '--disable-pip-version-check', 'install', '--prefer-binary', '-r', $requirementsPath) -Description 'Installing the pinned notebook dependencies (internet access required)'
    Invoke-PythonStep -Executable $venvPython -PythonArguments @('-m', 'pip', 'check') -Description 'Checking package dependency compatibility'

    $verifyCode = @'
import sys
sys.path.insert(0, sys.argv[1])
from notebook_setup import check_dependencies
check_dependencies()
import numpy, pandas, matplotlib, scipy, h5py
import tensorflow as tf
import keras, sciann, notebook, ipykernel
print('Scientific library imports passed.')
print('TensorFlow:', tf.__version__)
print('SciAnn:', sciann.__version__)
'@
    Invoke-PythonStep -Executable $venvPython -PythonArguments @('-c', $verifyCode, $codesPath) -Description 'Verifying the installed scientific libraries'
    Invoke-PythonStep -Executable $venvPython -PythonArguments @('-m', 'ipykernel', 'install', '--prefix', $venvPath, '--name', 'pinneikonal', '--display-name', 'PINNeikonal (Python 3.8)') -Description 'Registering the project Jupyter kernel'

    Write-Host "`nInstallation finished successfully."
    Write-Host "Open a codes\script*.ipynb notebook in VS Code and select PINNeikonal (Python 3.8), or select $venvPython as the Python environment."
    Write-Host 'Standalone Jupyter command:'
    Write-Host ('& "{0}" -m notebook "{1}"' -f $venvPython, $codesPath)
}
catch {
    [Console]::Error.WriteLine("Installation failed: " + $_.Exception.Message)
    exit 1
}
