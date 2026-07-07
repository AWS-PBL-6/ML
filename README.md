# AE-Sentinel ML / AI

SageMaker 학습/배포, 엔드포인트 런타임, 프롬프트 자산을 관리하는 저장소다.

## 역할

- 데이터 전처리 및 학습 파이프라인
- 모델 평가 및 등록
- 실시간 추론 엔드포인트 런타임
- Bedrock 프롬프트 자산 관리

## 구조

```text
training-pipeline/
  pipelines/
  jobs/
endpoint-runtime/
  src/
  tests/
prompt-assets/
  bedrock/
infra/
  ml-account/
```

## 계약 기준

- contracts canonical source: `AWS-PBL-6/docs`
- Backend 와의 추론 계약은 `inference-request.v1.json`, `inference-response.v1.json` 을 따른다.
