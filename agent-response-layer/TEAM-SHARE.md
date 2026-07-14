# [팀 공유] RAG Response Guidance

## 한 줄 구조

```text
Backend → Response Guidance Lambda → Bedrock RetrieveAndGenerate → 한국 공식 KB → JSON 반환
```

별도 Bedrock Agent, Agent Alias, Action Group은 사용하지 않는다. Backend는 Lambda를
`RequestResponse`로 호출해 반환 JSON을 이벤트에 저장한 뒤 WebSocket·SNS 이메일·등록 SMS에
공통으로 사용한다.

## 운영 리소스

| 항목 | 값 |
|---|---|
| Lambda | `ae-sentinel-app-response-guidance` |
| Knowledge Base | `JOCDF5OOGV` |
| Data Source | `AM44OYKZBB` |
| S3 포함 경로 | `approved/korean/` |
| 모델 | `apac.amazon.nova-lite-v1:0` inference profile |

## 운영 근거

- 해양수산부 `선박설비기준 별표 21의2`
- 한국해양교통안전공단 `해상교통안전진단·사전컨설팅 가이드(2023)`
- 해양수산부·수협중앙회 `연안선망어업 안전보건 표준매뉴얼`

입력 위험도는 Backend 확정값이므로 바꾸지 않는다. 입력에 없는 실장력·재질·MBL·스냅백
반경은 계산하거나 추정하지 않는다. RAG 실패 시 Lambda가 `confidence=LOW` 안전 폴백을
반환하며, 발송과 저장은 항상 Backend 책임이다.

## 운영 검증 결과 — 2026-07-14

- SAM 스택 `ae-sentinel-app` 배포 완료
- DANGER 테스트 HTTP/Lambda 상태 `200`, FunctionError 없음
- 응답 `source=KB`, `confidence=HIGH`, `alertDispatched=false`
- 실제 인용: `연안선망어업 안전보건 표준매뉴얼`, `선박설비기준 별표 21의2`
- 즉시 조치: 인원 이탈, 반동 위험구역 접근 통제, 작업 중지, 안전 확보 후 자격자 점검·보고
- 금지 조치인 임의 계류 해제·선박 이동 없음
- 최종 실행 이후 CloudWatch `ERROR` 0건
- 전용 Bedrock Agent/Alias/IAM 역할 및 해외·합성 KB 객체 삭제 완료
