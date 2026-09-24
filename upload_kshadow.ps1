# upload_kshadow.ps1
#
# Windows PowerShell version of upload_kshadow.sh. Pushes this folder to
# GitHub and tags a release, which is what Zenodo mints the DOI from.
#
# Run it from the folder that holds README.md and CITATION.cff:
#
#     powershell -ExecutionPolicy Bypass -File .\upload_kshadow.ps1 -User creelie
#
# Parameters:
#   -User    GitHub account      (default creelie)
#   -Repo    repository name     (default kshadow-coverage)
#   -Dir     folder to push      (default: the folder holding this script)
#   -Tag     release tag         (default v1.0.0)
#   -Private make the repository private (needs the gh CLI)
#
# Re-running it is safe: an existing git tree is updated, not wiped.

param(
    [string]$User = "creelie",
    [string]$Repo = "kshadow-coverage",
    [string]$Dir  = "",
    [string]$Tag  = "v1.0.0",
    [switch]$Private
)

function Fail($msg) { Write-Host $msg -ForegroundColor Red; exit 1 }

function Have($name) {
    $null = Get-Command $name -ErrorAction SilentlyContinue
    return $?
}

if (-not (Have git)) { Fail "git is not installed. Get it from https://git-scm.com/download/win" }

# Folder: the one holding this script when it looks like the repository.
if ($Dir -eq "") {
    $here = Split-Path -Parent $MyInvocation.MyCommand.Path
    if ((Test-Path (Join-Path $here "README.md")) -and (Test-Path (Join-Path $here "CITATION.cff"))) {
        $Dir = $here
    } else {
        $Dir = Join-Path (Get-Location) $Repo
    }
}
if (-not (Test-Path $Dir)) { Fail "No such folder: $Dir" }
Set-Location $Dir
if (-not (Test-Path "README.md")) { Fail "$Dir does not look like the repository (no README.md)." }

$Url = "https://github.com/$User/$Repo"

Write-Host "==> repository : $Url"
Write-Host "==> folder     : $(Get-Location)"
Write-Host "==> tag        : $Tag"
Write-Host ""

# 1. Point the placeholder URLs at the account actually being used.
if ($User -ne "creelie" -or $Repo -ne "kshadow-coverage") {
    foreach ($f in @("CITATION.cff", "README.md", ".zenodo.json")) {
        if (Test-Path $f) {
            $text = Get-Content $f -Raw
            if ($text -match "github\.com/creelie/kshadow-coverage") {
                $text = $text -replace "github\.com/creelie/kshadow-coverage", "github.com/$User/$Repo"
                Set-Content -Path $f -Value $text -NoNewline
                Write-Host "    rewrote URL in $f"
            }
        }
    }
}

# 2. Clear out build litter that .gitignore excludes anyway.
Get-ChildItem -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
    ForEach-Object { Remove-Item $_.FullName -Recurse -Force -ErrorAction SilentlyContinue }
Get-ChildItem -Recurse -File -Filter "*.pyc" -ErrorAction SilentlyContinue |
    Remove-Item -Force -ErrorAction SilentlyContinue

# 3. Refuse anything GitHub will reject on size.
$big = Get-ChildItem -Recurse -File | Where-Object {
    $_.Length -gt 90MB -and $_.FullName -notmatch "\\\.git\\"
}
if ($big) {
    Write-Host "These files are over 90 MB and need Git LFS:" -ForegroundColor Red
    $big | ForEach-Object { Write-Host ("    " + $_.FullName) }
    exit 1
}

# 4. Local repository.
if (-not (Test-Path ".git")) {
    git init -q
    git branch -M main
}

$gitName  = (git config user.name)  2>$null
$gitEmail = (git config user.email) 2>$null
if ([string]::IsNullOrWhiteSpace($gitName) -or [string]::IsNullOrWhiteSpace($gitEmail)) {
    Write-Host "Git has no identity set. Run these once, then run this script again:"
    Write-Host '    git config --global user.name  "Deep Bhattacharjee"'
    Write-Host '    git config --global user.email "you@example.com"'
    exit 1
}

git add -A
git diff --cached --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "    nothing new to commit"
} else {
    git commit -q -m ("k-shadow coverage certificates, " + $Tag.TrimStart("v"))
    if ($LASTEXITCODE -ne 0) { Fail "commit failed" }
    Write-Host "    committed"
}

# 5. Remote. gh creates the repository; without gh, create it in the browser first.
git remote get-url origin 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) {
    git remote set-url origin "$Url.git"
} else {
    if (Have gh) {
        gh auth status 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) {
            gh repo view "$User/$Repo" 2>$null | Out-Null
            if ($LASTEXITCODE -ne 0) {
                $vis = "--public"
                if ($Private) { $vis = "--private" }
                gh repo create "$User/$Repo" $vis --description "Topological certificates of redundant coverage for cortical electrode arrays"
            }
        }
    }
    git remote add origin "$Url.git"
}

# 6. Push, then tag.
git push -u origin main
if ($LASTEXITCODE -ne 0) { Fail "push failed. Create $Url in the browser first, or sign in with: gh auth login" }

git rev-parse $Tag 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "    tag $Tag already exists locally"
} else {
    git tag -a $Tag -m "Version accompanying the manuscript"
}
git push origin $Tag

Write-Host ""
Write-Host "Done. $Url"
Write-Host "Next: switch the repository on at zenodo.org/account/settings/github,"
Write-Host "then re-push the tag (or draft a release) so Zenodo mints the DOI."
