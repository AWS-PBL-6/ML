# SageMaker ML 계정 배포 자산

계류삭 연결 고리 피에조 데이터의 실제 SageMaker 실행 코드다.

- `processing.py`: 원천 CSV 계약·라벨을 검증하고 세션/시간순 학습 입력을 출력한다.
- `train.py`: Processing 출력을 받아 세션 GroupKFold 윈도 XGBoost를 학습하고 `/opt/ml/model`에 서빙 코드를 포함한 모델을 저장한다.
- 학습 원본: `ml/training-pipeline/port_breakage_pipeline.py`, `port_window_features.py`
- 추론 코드: `ml/endpoint-runtime/sagemaker/inference.py`

클라우드 자산 이름:

- S3: `s3://ae-sentinel-sagemaker-727847798739/line-eye/`
- Processing/Training prefix: `ae-sentinel-line-eye-*`
- Model Package Group: `ae-sentinel-line-eye-models`
- Endpoint: `ae-sentinel-port-window`

모델 패키지는 `PendingManualApproval`로 등록한 뒤 검증 성공 시 `Approved`로 변경하고,
승인된 모델만 기존 serverless endpoint에 배포한다.
