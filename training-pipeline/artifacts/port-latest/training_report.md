# Port-domain AE breakage model — ae-port-window-xgb-v1-20260709

- 생성: 2026-07-09T08:04:55.000Z  · 데이터: port_bollard_ae_dataset.csv
- 윈도: 직전 12개 이벤트 (최소 3개) · 세션 55개 · 학습 윈도 35403행
- 모델: XGBoost 3-class(multi:softprob) + 온도 스케일링 T=2.48

## OOF 성능 (GroupKFold 5, 세션 단위 · 배포 결정규칙=severity 임계)

| 정확도 | macro-F1 | 위험 재현율 | 위험 정밀도 |
|---|---|---|---|
| 0.8727 | 0.8758 | 0.97 | 1.0 |

혼동행렬(행=실제 N/C/D, 열=예측): `[[12449, 1409, 0], [2784, 8305, 0], [20, 294, 10142]]`

## 임계 정책

- warnThreshold(severity) = 0.3224
- dangerThreshold(severity) = 0.9572
- severity = 0.5·P(주의) + 1.0·P(위험)

## 배포

`LOCAL_ML_MODEL_PATH`를 이 디렉토리의 `model_artifact.json`으로 지정하면 백엔드
수정 없이 로컬 추론에 연결된다. 서빙 측은 `repo.list_events(lineId, limit=window.size)`로
직전 이벤트를 모아 `featureSpec.runtimeFeatureNames` 순서의 윈도 벡터를 만든다
(로직: `port_window_features.window_vector`).
