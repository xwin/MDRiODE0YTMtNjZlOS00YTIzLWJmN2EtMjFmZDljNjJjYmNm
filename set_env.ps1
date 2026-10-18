# Sets PYTHONPATH to the absolute path of this project root
$projectRoot = $PSScriptRoot
if (-not $projectRoot) {
    $projectRoot = "D:\git\sammy\ecs170\ECS170_Fall_2026_Source_Code_Template"
}

$env:PYTHONPATH = $projectRoot
Write-Host "PYTHONPATH set to: $env:PYTHONPATH"
