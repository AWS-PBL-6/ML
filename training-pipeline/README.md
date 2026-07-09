# training-pipeline

SageMaker Processing, Training, Evaluation, Registry 관련 코드를 둔다.

## Local PoC pipeline

현재 저장소에는 proxy dataset 기반의 로컬 학습 파이프라인이 포함된다.

- 입력: Kaggle `AE_Damage_Detection_Dataset.csv`
- core labels: `0` vs `2`
- shadow labels: `1`, `3`
- feature set: `peakAmplitude`, `energy`, `durationMs`, `riseTimeMs`,
  `dominantFrequencyHz`, `temperatureC`, `humidityPct`
- auto search:
  - `baseline` vs `engineered` feature set 비교
  - hybrid weight / OOD penalty 후보 비교
  - shadow spill-over를 함께 보는 champion selection
- artifact 구조:
  - core logistic breakage model
  - class exemplar vectors
  - similarity / OOD calibration 값
  - runtime threshold / temporal policy 값
  - local demo profile
- 출력:
  - `prepared_core_dataset.csv`
  - `prepared_shadow_dataset.csv`
  - `model_artifact.json`
  - `training_summary.json`
  - `training_report.md`

실행 예시:

```bash
python3 ml/training-pipeline/proxy_breakage_pipeline.py \
  --csv /Users/mac/Downloads/AE_Damage_Detection_Dataset.csv
```

기본 출력 디렉토리:

```text
ml/training-pipeline/artifacts/latest/
```

`CatBoost`, `LightGBM`, `XGBoost`가 설치되어 있으면 비교 실험을 수행하고,
없으면 dependency-free hybrid artifact를 생성한다. 현재 hybrid score는
`0 vs 2` logistic score, danger-side similarity, OOD penalty를 조합한다.
