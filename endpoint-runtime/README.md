# endpoint-runtime

SageMaker Endpoint 추론 런타임 코드를 둔다.

## Local runtime

`local_model_runtime.py`는 `training-pipeline`이 생성한 `model_artifact.json`을
읽어 `inference-request.v1` -> `inference-response.v1` 형식으로 추론한다.

이 런타임은 두 용도로 쓴다.

1. SageMaker 없이 로컬 artifact 검증
2. `backend/application-api`의 `LOCAL_ML_MODEL_PATH` 경로로 백엔드와 직접 연결

현재 로컬 artifact가 hybrid 형식이면 런타임은 아래를 함께 처리한다.

- `breakageScore`
- `dangerSimilarity` / `normalSimilarity`
- `oodScore`
- raw 7개 입력으로부터 engineered feature 자동 파생
- policy threshold 기반 `riskLevel`
- `explain_request(...)`를 통한 최근접 danger/normal exemplar 조회
