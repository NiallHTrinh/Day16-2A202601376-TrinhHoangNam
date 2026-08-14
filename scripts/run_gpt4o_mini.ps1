# Smoke-test the harness against OpenAI gpt-4o-mini.
# Prerequisites:
#   1. copy .env.example .env
#   2. paste ARENA_API_KEY=sk-... into .env
#
# Usage:
#   .\scripts\run_gpt4o_mini.ps1
#   .\scripts\run_gpt4o_mini.ps1 -Brief pub-08-an-toan-boc-do
#   .\scripts\run_gpt4o_mini.ps1 -All

param(
    [string]$Brief = "pub-01-sla-hien-hanh",
    [switch]$All
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path ".env")) {
    Write-Host "Chưa có .env — copy từ mẫu rồi dán API key:" -ForegroundColor Yellow
    Write-Host "  copy .env.example .env"
    Write-Host "  notepad .env"
    exit 1
}

$argsList = @(
    "scripts/run_practice.py",
    "--model", "real",
    "--prompt-addendum",
    "--layers", "all",
    "--entry", "gpt4o-mini",
    "--out", "runs/gpt4o-mini.json",
    "--max-tokens", "2048"
)

if (-not $All) {
    $argsList += @("--brief", $Brief)
}

Write-Host "Running: python $($argsList -join ' ')" -ForegroundColor Cyan
python @argsList
if ($LASTEXITCODE -eq 0) {
    python scripts/selfeval.py --run runs/gpt4o-mini.json --summary
}
