# Push this repo to GitHub.
# 1. Create a new repo on https://github.com/new (name e.g. LLM_Benchmarking), leave it empty.
# 2. Edit the two lines below with YOUR GitHub username and repo name, then run:
#    .\push_to_github.ps1
# Or run: .\push_to_github.ps1 -Username "yourusername" -RepoName "LLM_Benchmarking"

param(
    [string]$Username = "YOUR_GITHUB_USERNAME",
    [string]$RepoName = "LLM_Benchmarking"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if ($Username -eq "YOUR_GITHUB_USERNAME") {
    $Username = Read-Host "Enter your GitHub username"
}
if ([string]::IsNullOrWhiteSpace($Username)) {
    Write-Error "Username is required."
}
if ([string]::IsNullOrWhiteSpace($RepoName)) {
    $RepoName = Read-Host "Enter repo name (default: LLM_Benchmarking)"
    if ([string]::IsNullOrWhiteSpace($RepoName)) { $RepoName = "LLM_Benchmarking" }
}

$url = "https://github.com/$Username/$RepoName.git"
Write-Host "Remote: $url" -ForegroundColor Cyan

$remotes = git remote 2>$null
if ($remotes -match "origin") {
    git remote set-url origin $url
    Write-Host "Updated existing remote 'origin'."
} else {
    git remote add origin $url
    Write-Host "Added remote 'origin'."
}

git branch -M main
Write-Host "Pushing to origin main..." -ForegroundColor Green
git push -u origin main
Write-Host "Done. Open https://github.com/$Username/$RepoName" -ForegroundColor Green
