# Proxy Breakage Training Report

- Generated at: `2026-07-09T03:47:50+00:00`
- Source CSV: `/Users/mac/Downloads/AE_Damage_Detection_Dataset.csv`
- Model version: `ae-proxy-breakage-hybrid-v2`
- Selected feature set: `engineered`
- Features: `peakAmplitude, energy, durationMs, riseTimeMs, dominantFrequencyHz, temperatureC, humidityPct, logEnergy, energyPerDuration, riseRatio, freqXRise, tempXHumidity`
- Hybrid weights: `breakage=0.60`, `similarity=0.40`, `oodPenalty=0.30`

## Core dataset

- `Damage_Type=0`: `418` rows
- `Damage_Type=2`: `200` rows

## Shadow dataset

- `Damage_Type=1`: `288` rows
- `Damage_Type=3`: `94` rows

## Core CV metrics @ threshold 0.5

- Accuracy: `0.5469`
- Precision: `0.3347`
- Recall: `0.4050`
- F1: `0.3665`
- FNR: `0.5950`
- ROC-AUC: `0.4985`
- PR-AUC: `0.3464`

## Threshold policy

- `T_warn`: `0.2743`
- `T_danger`: `0.3743`
- `T_warn_release`: `0.1743`
- `T_danger_release`: `0.2743`
- `OOD caution`: `0.50`
- `OOD block danger`: `0.75`
- Runtime guardrail: inputs are clipped to the core-training feature range before scoring.

### Danger threshold metrics

- Precision: `0.3256`
- Recall: `0.9850`
- F1: `0.4894`
- FNR: `0.0150`

## Shadow score behavior

- Matrix Cracking mean score: `0.4850`
- Matrix Cracking >= warn ratio: `1.0000`
- Matrix Cracking >= danger ratio: `0.9965`
- Delamination mean score: `0.4890`
- Delamination >= warn ratio: `1.0000`
- Delamination >= danger ratio: `0.9894`

## Feature-set search

- Candidate count: `12`
- Selected `PR-AUC`: `0.3464`
- Selected shadow danger mean ratio: `0.9929`

## Optional booster comparisons

- `catboost` status: `ok`
  precision=0.3483, recall=0.5050, f1=0.4122, roc_auc=0.5518, pr_auc=0.3479
- `lightgbm` status: `ok`
  precision=0.3246, recall=0.4350, f1=0.3718, roc_auc=0.5274, pr_auc=0.3292
- `xgboost` status: `ok`
  precision=0.3650, recall=0.4800, f1=0.4147, roc_auc=0.5469, pr_auc=0.3556

## Augmentation roadmap

- Do not oversample before the core 0-vs-2 baseline is stable.
- First priority is rope real-data capture, not synthetic tabular inflation.
- If augmentation is required later, prefer waveform-level perturbations or conservative feature-space jitter bounded by the real sensor range.
- Keep labels `1` and `3` as shadow sets unless real rope labels support a broader target definition.
