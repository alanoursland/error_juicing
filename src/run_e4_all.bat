@echo off
REM All 18 E4 runs (6 arms x 3 seeds), sequential, resumable. Run on the local
REM GPU from the repo root:
REM   run_e4_all.bat
REM A killed run loses at most one epoch; rerunning this script resumes.
REM Each run prints its wall-clock estimate at startup. Commit results/e4/*.json
REM afterwards; figures regenerate via: python src/fig_e4.py

for %%s in (0 1 2) do (
  for %%a in (baseline weight_decay label_smoothing logitnorm focal temperature) do (
    python src/e4_fixes.py --config "configs/e4/%%a.yaml" --seed "%%s" --out results/e4/
    
    REM This mimics the 'set -e' behavior by stopping the script if Python throws an error
    if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
  )
)