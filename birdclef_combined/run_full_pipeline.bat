@echo off
REM =============================================================================
REM run_full_pipeline.bat — Complete BirdCLEF 2025 Training Pipeline
REM =============================================================================
REM Edit the DATA_ROOT and PSEUDO_DIR paths below before running.
REM Run this from the birdclef_combined\ directory.
REM
REM Usage:
REM   cd C:\Users\User\Downloads\birdclef_2025_combined_FINAL\birdclef_combined
REM   run_full_pipeline.bat
REM =============================================================================

SET DATA_ROOT=D:\birdclef_data\birdclef-2025
SET SOUNDSCAPE_DIR=%DATA_ROOT%\train_soundscapes
SET PSEUDO_DIR=D:\birdclef_data\pseudo_labels
SET LOGDIR=logdir
SET PYTHON=python

REM Activate your virtual environment
CALL ..\birdclef_env\Scripts\activate.bat

ECHO.
ECHO ============================================================
ECHO  STAGE 1: Supervised Training (ECA-NFNet-L0, all 5 folds)
ECHO ============================================================
%PYTHON% scripts/main_train.py configs/train/selected_eca.py
IF %ERRORLEVEL% NEQ 0 GOTO ERROR

ECHO.
ECHO ============================================================
ECHO  STAGE 2: Supervised Training (EfficientNetV2-S, all 5 folds)
ECHO ============================================================
%PYTHON% scripts/main_train.py configs/train/selected_ebs.py
IF %ERRORLEVEL% NEQ 0 GOTO ERROR

ECHO.
ECHO ============================================================
ECHO  STAGE 3: Novel Components (Calibration + GCN)
ECHO  This takes ~30 minutes and must run AFTER Stage 1+2
ECHO ============================================================
FOR /F "delims=" %%i IN ('dir /b /s %LOGDIR%\eca_nfnet*\checkpoint_manifest.json') DO SET ECA_MANIFEST=%%i
%PYTHON% scripts/train_novel_components.py ^
    --manifest "%ECA_MANIFEST%" ^
    --data_root "%DATA_ROOT%" ^
    --soundscape_root "%SOUNDSCAPE_DIR%" ^
    --output_dir "%LOGDIR%\novel_components" ^
    --backbone eca_nfnet_l0
IF %ERRORLEVEL% NEQ 0 GOTO ERROR

ECHO.
ECHO ============================================================
ECHO  STAGE 4A: Generate Pseudo-Labels — Iteration 1
ECHO  Teacher: Stage 1 ECA (5 folds). Power alpha=0.6
ECHO  Expected: ~19,000 soundscape chunks selected
ECHO ============================================================
%PYTHON% scripts/generate_pseudo_labels.py ^
    --checkpoint_manifest "%ECA_MANIFEST%" ^
    --soundscape_dir "%SOUNDSCAPE_DIR%" ^
    --output_dir "%PSEUDO_DIR%" ^
    --iteration 1 ^
    --power_alpha 0.6 ^
    --backbone eca_nfnet_l0 ^
    --mode full
IF %ERRORLEVEL% NEQ 0 GOTO ERROR

ECHO.
ECHO ============================================================
ECHO  STAGE 4B: Pseudo Training — Iteration 1 (ECA-NFNet)
ECHO  Student trains on 50%% real + 50%% pseudo data
ECHO ============================================================
%PYTHON% scripts/main_train.py configs/train/pseudo_iter1_eca.py
IF %ERRORLEVEL% NEQ 0 GOTO ERROR

ECHO.
ECHO ============================================================
ECHO  STAGE 5A: Generate Pseudo-Labels — Iteration 2
ECHO  Teacher: Stage 4B ECA Student. Power alpha=0.5
ECHO  Mode: OOF + Full (merge both)
ECHO ============================================================
FOR /F "delims=" %%i IN ('dir /b /s %LOGDIR%\eca_nfnet*PseudoF2PT05MT01P06I1*\checkpoint_manifest.json') DO SET ECA_PSEUDO1_MANIFEST=%%i

%PYTHON% scripts/generate_pseudo_labels.py ^
    --checkpoint_manifest "%ECA_PSEUDO1_MANIFEST%" ^
    --soundscape_dir "%SOUNDSCAPE_DIR%" ^
    --output_dir "%PSEUDO_DIR%" ^
    --iteration 2 ^
    --power_alpha 0.5 ^
    --backbone eca_nfnet_l0 ^
    --mode full
IF %ERRORLEVEL% NEQ 0 GOTO ERROR

%PYTHON% scripts/generate_pseudo_labels.py ^
    --checkpoint_manifest "%ECA_PSEUDO1_MANIFEST%" ^
    --soundscape_dir "%SOUNDSCAPE_DIR%" ^
    --output_dir "%PSEUDO_DIR%" ^
    --iteration 2 ^
    --power_alpha 0.5 ^
    --backbone eca_nfnet_l0 ^
    --mode oof
IF %ERRORLEVEL% NEQ 0 GOTO ERROR

REM Merge iter2 full + oof
%PYTHON% scripts/merge_pseudo_labels.py ^
    --inputs "%PSEUDO_DIR%\pseudo_labels_iter2_full.parquet" ^
             "%PSEUDO_DIR%\pseudo_labels_iter2_oof.parquet" ^
    --output "%PSEUDO_DIR%\pseudo_labels_iter2_combined.parquet"
IF %ERRORLEVEL% NEQ 0 GOTO ERROR

ECHO.
ECHO ============================================================
ECHO  STAGE 5B: Pseudo Training — Iteration 2 (EfficientNetV2)
ECHO ============================================================
%PYTHON% scripts/main_train.py configs/train/pseudo_iter2_ebs.py
IF %ERRORLEVEL% NEQ 0 GOTO ERROR

ECHO.
ECHO ============================================================
ECHO  STAGE 6A: Generate Pseudo-Labels — Iteration 3
ECHO  Teacher: Stage 5B EBS Student. Power alpha=0.4
ECHO ============================================================
FOR /F "delims=" %%i IN ('dir /b /s %LOGDIR%\tf_efficientnetv2*PseudoF2PT05MT01P05I2*\checkpoint_manifest.json') DO SET EBS_PSEUDO2_MANIFEST=%%i

%PYTHON% scripts/generate_pseudo_labels.py ^
    --checkpoint_manifest "%EBS_PSEUDO2_MANIFEST%" ^
    --soundscape_dir "%SOUNDSCAPE_DIR%" ^
    --output_dir "%PSEUDO_DIR%" ^
    --iteration 3 ^
    --power_alpha 0.4 ^
    --backbone tf_efficientnetv2_s ^
    --mode full
IF %ERRORLEVEL% NEQ 0 GOTO ERROR

REM Merge all 3 iterations together for final training
%PYTHON% scripts/merge_pseudo_labels.py ^
    --inputs "%PSEUDO_DIR%\pseudo_labels_iter1_full.parquet" ^
             "%PSEUDO_DIR%\pseudo_labels_iter2_combined.parquet" ^
             "%PSEUDO_DIR%\pseudo_labels_iter3_full.parquet" ^
    --output "%PSEUDO_DIR%\pseudo_labels_all_iters_merged.parquet"
IF %ERRORLEVEL% NEQ 0 GOTO ERROR

ECHO.
ECHO ============================================================
ECHO  STAGE 6B: Pseudo Training — Iteration 3 (ECA-NFNet, FINAL)
ECHO ============================================================
%PYTHON% scripts/main_train.py configs/train/pseudo_iter3_eca.py
IF %ERRORLEVEL% NEQ 0 GOTO ERROR

ECHO.
ECHO ============================================================
ECHO  STAGE 7: Generate Final Submission
ECHO ============================================================
%PYTHON% scripts/main_inference.py configs/inference/ensemble_final.py --mode submit
IF %ERRORLEVEL% NEQ 0 GOTO ERROR

ECHO.
ECHO ============================================================
ECHO  ALL STAGES COMPLETE
ECHO  Check logdir\ for checkpoints and submission.csv
ECHO ============================================================
GOTO END

:ERROR
ECHO.
ECHO [ERROR] Pipeline failed at the step above. Check the error message.
ECHO Resume from where it stopped by running the failed command manually.
EXIT /B 1

:END
