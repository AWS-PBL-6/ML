# training-pipeline

SageMaker Processing, Training, Evaluation, Registry 관련 코드를 둔다.

## 현행: 홋줄 AE 데이터셋 (논문 기반 → 항만 도메인 변환)

계류삭 AE 파단 실측 논문(Bashir et al. 2017, *Applied Acoustics* 121)을 근거로
한 합성 데이터셋을 사용한다. 근거·평가·검증 문서는
[docs/docs](../../docs/docs) 의 `rope-ae-dataset-evaluation.md`,
`port-ae-dataset.md`, `dataset-reliability-report.md` 참조.

### 항만 도메인 변환 생성기 — `port_domain_synthesis.py`

수중 하이드로폰 계측(dB re 1 μPa) 기반 합성 데이터셋을 **안벽 볼라드 부착
접촉식 AE 센서 + 실항만 배경 소음** 도메인으로 변환한다. 단위 변환·거리 감쇠·
부동 검출 임계·크레인/강우/바람 교란 주입을 수행하며, 모든 가정 파라미터는
파일 상단 `PARAMS` 한 곳에 집약되어 실센서 수집 후 재보정할 수 있다.

```bash
PYTHONPATH=<repo>/.vendor_ml python3 port_domain_synthesis.py \
  --src ~/Downloads/synthetic_rope_damage_classification_30k.csv \
  --out ~/Downloads/port_bollard_ae_dataset.csv
```

### 학습 원칙 (검증됨)

- 입력은 개별 이벤트가 아니라 **직전 N개 이벤트의 시계열 윈도 집계**
  (신호군 조성·이벤트율·진폭/히트 통계)
- 하중비·절대시간·시험기 좌표(`Truth_*`)는 라벨 누수·배포 불가 정보 → 학습 제외
- 분할은 **GroupKFold(Session 단위)**, 평가지표는 **위험 클래스 재현율 + 혼동행렬**
- 검증 결과(XGBoost, GroupKFold 5): 윈도 집계 + 환경 피처 정확도 87.3%,
  위험 재현율 98.9% / 정밀도 100%

### 정식 학습 파이프라인 — `port_breakage_pipeline.py` (구현됨)

`port_domain → 윈도 피처 → XGBoost 3-class → 온도 스케일링 보정 → 아티팩트`.
산출 아티팩트는 `local_model_runtime.py`의 `port_window_xgb_v1` 경로가 읽으며,
백엔드 수정 없이 `LOCAL_ML_MODEL_PATH`로 연결된다.

```bash
PYTHONPATH=<repo>/.vendor_ml python3 port_breakage_pipeline.py \
  --src ~/Downloads/port_bollard_ae_dataset.csv
# → artifacts/port-latest/{model_artifact.json, port_window_xgb.json, training_report.md}
```

- 윈도 피처 계약: `port_window_features.py` (`window_vector`, `WINDOW_FEATURES`,
  `WINDOW_SIZE=12`) — 서빙 측(백엔드 inference_service)이 동일 로직으로 조립
- riskScore = 기대 심각도(0.5·P(주의)+1.0·P(위험)) → 백엔드 riskScore-임계 히스테리시스와 정합
- OOF(GroupKFold 5, 세션): 정확도 0.873 · macro-F1 0.876 · **위험 재현율 0.97 / 정밀도 1.00**
- 아티팩트에 `featureSpec.runtimeFeatureNames`(이름·순서)·`modelVersion`·`policy` 동봉

### 예정: SageMaker 엔드포인트 전환

1단계는 λ 내장 추론(위 아티팩트, `LOCAL_ML_MODEL_PATH`). 2단계는 동일 부스터를
SageMaker 엔드포인트로 서빙(`ml/infra`) — 백엔드는 `SAGEMAKER_ENDPOINT`만 설정하면 전환.

## 보관: Kaggle proxy 파이프라인 (레거시)

> Kaggle 항공기 복합재 AE 데이터셋 기반의 `0 vs 2` proxy 접근. 현행 방법이 아니며
> 배경은 [보관 문서](../../docs/docs/archive/README.md) 참조. 코드/아티팩트는 로컬
> 추론 경로(`LOCAL_ML_MODEL_PATH`)에 물려 있어, 새 rope-AE 파이프라인이 동일
> 아티팩트 계약으로 대체할 때까지 남겨둔다.

- 스크립트: `proxy_breakage_pipeline.py`
- 입력: Kaggle `AE_Damage_Detection_Dataset.csv` (core `0` vs `2`, shadow `1`/`3`)
- 산출물: `artifacts/latest/model_artifact.json`, `training_summary.json`,
  `training_report.md`, `prepared_core_dataset.csv`, `prepared_shadow_dataset.csv`

```bash
python3 proxy_breakage_pipeline.py --csv /Users/mac/Downloads/AE_Damage_Detection_Dataset.csv
```
