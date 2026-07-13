# Backend ↔ agent 인터페이스

> 분류: 현행 협업 계약
> 기준일: 2026-07-13
> 상태: 현행(Canonical)
> 계약 소스: `ML/agent-response-layer/contracts/`
> - 입력: `schemas/agent-request.v1.json` (예제 `examples/agent-request.v1.json`)
> - 출력: `schemas/response-guidance.v1.json` (예제 `examples/response-guidance.v1.json`)

이 문서는 Backend와 agent(Bedrock 대응 계층) 사이에 주고받는 데이터와 책임을 정의한다.
필드의 기계 검증 기준은 위 JSON Schema가 우선하며, 이 문서는 그 계약을 실제 구현 순서로 설명한다.

## 1. 한 문장 원칙

Backend가 위험판정(`CAUTION`/`DANGER`)을 확정하면 agent를 내부 Invoke하여, Knowledge
Base(RAG)에서 상황·항만사별 대처방안을 생성해 받는다. **agent는 위험도를 재판정하거나 물리
수치(스냅백 반경 등)를 계산하지 않고**, 결정론적 결과를 관리자용 설명 문장과 UI 친화 JSON으로
가공만 한다. 발송(SNS·WebSocket)과 저장은 Backend 책임이다.

```
IoT → Backend → SageMaker(추론) → Backend
                                    └─[CAUTION/DANGER]─▶ agent (Bedrock KB RAG)
                                                            │ response-guidance 반환
                                    Backend ◀───────────────┘
                                      ├─ EventRecord 저장(guidance)
                                      ├─ SNS 발송 (관리자)
                                      └─ WebSocket push → 대시보드
```

> agent는 ML과 직접 통신하지 않는다. ML 결과(`inference-response.v1`)는 Backend가 받아
> 위험도를 확정한 뒤, Backend가 agent를 호출한다. 즉 경계는 **Backend → agent → Backend**.

## 2. 경계 책임

| 주체 | 책임 | 하지 않는 것 |
|---|---|---|
| Backend | agent 호출, 입력 조립, guidance 저장(EventRecord), SNS·WebSocket 발송 | 대처방안 문장 직접 생성 |
| agent | KB 검색 + 대처방안 생성(설명·UI JSON) | 발송·저장, 위험도 재판정, 물리 수치 계산 |

## 3. 호출 방식

- **내부 Lambda Invoke** (`RequestResponse` 권장). 공개 REST가 아니므로 **Cognito 불필요.**
- 대상 함수: `ae-sentinel-response-guidance`
- 트리거 조건: `riskLevel ∈ {CAUTION, DANGER}`. `NORMAL`은 호출 생략 권장(호출해도 agent가
  알림을 억제하고 immediateActions를 비운다).

## 4. 입력 계약 (Backend → agent) — `agent-request.v1`

| 필드 | 타입 | 필수 | 설명 |
|---|---|---|---|
| `schemaVersion` | `"1.0.0"` | ✅ | 계약 버전 |
| `traceId` | string | ✅ | 저장/추적 기준 키 |
| `eventId` | string | ✅ | 중복 처리 기준 키 |
| `requestId` | string | | 호출 상관관계 |
| `lineId` | string | ✅ | 진단 대상 계류삭(센서 귀속처) |
| `siteId` | string | ✅ | 사이트(항만사 규정 우선 판단에 사용) |
| `berthId` | string | | 선석(현장 위치 안내용) |
| `bollardId` | string | | 안벽 연결 위치(위치 안내용, 센서 소유자 아님) |
| `vesselId` | string | | 접안 세션 선박(없으면 미계선) |
| `riskLevel` | enum `NORMAL|CAUTION|DANGER` | ✅ | **확정값**. agent가 바꾸지 않음 |
| `riskScore` | number 0–1 | | 위험 점수 |
| `classProbabilities` | object | | NORMAL/CAUTION/DANGER 확률 |
| `modelVersion` | string | | 추론 모델 버전 |
| `features` | object | | 근거 수치. 아키텍처 §6.2 어휘(아래). 추가 키 허용 |
| `context` | object | | 재질/장력/환경. 추가 키 허용 |

- `features`: `rms`, `peakAmplitude`, `energy`, `dominantFrequencyHz`, `riseTimeMs`, `durationMs`
- `context`: `materialType`, `tensionEstimateKn`, `temperatureC`, `humidityPct`,
  (피에조 경로) `sensorKind`, `structuralFreqHz`, `loadRatio`, `windMps`, `craneActive`

### 입력 예시
```json
{
  "schemaVersion": "1.0.0",
  "traceId": "trc-20260713-0005",
  "eventId": "evt-20260713-0005",
  "lineId": "mooring-05",
  "siteId": "yard-a",
  "berthId": "berth-01",
  "bollardId": "bollard-05",
  "vesselId": "vessel-02",
  "riskLevel": "DANGER",
  "riskScore": 0.962,
  "classProbabilities": { "NORMAL": 0.012, "CAUTION": 0.026, "DANGER": 0.962 },
  "modelVersion": "port-window-xgb-v1",
  "features": { "rms": 0.82, "peakAmplitude": 0.96, "energy": 1450,
                "dominantFrequencyHz": 210000, "riseTimeMs": 3.2, "durationMs": 48 },
  "context": { "materialType": "synthetic-rope", "tensionEstimateKn": 168,
               "temperatureC": 26.1, "humidityPct": 78,
               "sensorKind": "piezo", "structuralFreqHz": 30.8, "loadRatio": 0.93 }
}
```

## 5. 출력 계약 (agent → Backend) — `response-guidance.v1`

| 필드 | 타입 | 설명 |
|---|---|---|
| `schemaVersion` | `"1.0.0"` | 계약 버전 |
| `type` | `"response.guidance.generated"` | 메시지 종류 |
| `traceId` / `eventId` | string | 입력에서 전달된 값 유지 |
| `generatedAt` | date-time(UTC) | 생성 시각 |
| `lineId` / `vesselId` / `siteId` | string | 대상 식별 |
| `riskLevel` | enum | 입력값 그대로(불변) |
| `riskScore` | number 0–1 | 입력값 유지 |
| `source` | `KB` \| `AGENT` | 생성 경로(RAG 직접 / Bedrock Agent) |
| `alertDispatched` | boolean | agent 자체 SNS 발송 여부(통합 시 false 권장) |
| `guidance` | object | 아래 대처방안 객체 |

`guidance` 객체:

| 필드 | 타입 | 설명 |
|---|---|---|
| `riskLevel` | enum | 위험도 |
| `headline` | string(≤90자) | 관리자용 한 줄 요약(SMS 가능) |
| `situation` | string | 현재 상황 2~3문장 |
| `immediateActions` | string[] | 즉시 조치(NORMAL이면 빈 배열) |
| `followupActions` | string[] | 후속 조치 |
| `citations` | string[] | 참조한 규정 문서·항목(RAG 근거) |
| `confidence` | `HIGH` \| `MEDIUM` \| `LOW` | 근거 충분성 |

### 출력 예시
```json
{
  "schemaVersion": "1.0.0",
  "type": "response.guidance.generated",
  "traceId": "trc-20260713-0005",
  "eventId": "evt-20260713-0005",
  "generatedAt": "2026-07-13T09:00:01.480Z",
  "lineId": "mooring-05",
  "vesselId": "vessel-02",
  "siteId": "yard-a",
  "riskLevel": "DANGER",
  "riskScore": 0.962,
  "source": "KB",
  "alertDispatched": false,
  "guidance": {
    "riskLevel": "DANGER",
    "headline": "5번 계류삭 위험. 데드존 통제·하역 중단·인원 이탈 확인",
    "situation": "5번 계류삭의 AE 진폭·에너지가 급증하고 고주파 성분이 관측되었습니다. 합성섬유 로프 추정 장력이 MBL의 70%를 초과해 파단·스냅백 위험이 높습니다.",
    "immediateActions": [
      "스냅백 위험구역(데드존) 내 인원 이탈 완료 확인",
      "해당 선석 크레인·하역 작업 즉시 중단",
      "위험구역 접근 통제"
    ],
    "followupActions": [
      "선박·도선사에 계류 재조정 요청",
      "인접 볼라드 라인 연쇄 파단 가능성 점검",
      "비상 보고 체계 가동"
    ],
    "citations": [
      "부산신항 A터미널 계류 안전 운영규정 §4 DANGER 사내 대처 절차",
      "계류삭 안전관리 일반지침 §3 DANGER 표준 대처방안"
    ],
    "confidence": "HIGH"
  }
}
```

## 6. Backend 저장·발송 규칙

- **저장**: `guidance` 객체를 이벤트에 보존한다.
  - 권장: `EventRecord`에 `responseGuidance` 객체 필드를 추가해 그대로 저장(대시보드가 구조적
    렌더링). — 계약 변경(백엔드 담당) 필요.
  - 임시(무변경): `EventRecord.alertMessage`에 `guidance.headline`을 넣고, 전체 객체는
    자유형 `EventRecord.context`에 보관.
- **SNS**: 관리자 알림 `message` = `guidance.headline`(+ `immediateActions` 요약). 발송은
  Backend `app-dispatch-alert`가 담당. **agent 자체 SNS는 통합 시 비활성화**(`SNS_TOPIC_ARN`
  공란)하여 중복 발송을 막는다.
- **WebSocket**: `risk.danger.detected` / `risk.caution.detected`의 `data.alertMessage`에
  `guidance.headline`을 실어 대시보드로 push. 상세(즉시/후속/근거)는 REST 조회로 표시.

## 7. 처리 규칙 · 지식 출처

- **지식 출처 2종**: ① 공개 일반 안전매뉴얼(`scope=general`) ② 항만사 개별 규정
  (`scope=port-specific`, `siteId` 일치 시 일반 지침보다 **우선 적용**).
- **불변식**: 위험도 확정값 유지 · 물리 수치 재계산 금지 · 대처방안은 KB 검색 근거 기반
  (지어내기 금지) · 현장 작업자 대피 경보는 별도 계층(agent는 **관리자** 대상 안내).

## 8. 오류 · 폴백

- Bedrock 호출 실패 시에도 agent는 죽지 않고 **fallback 템플릿**으로 위험도별 최소 안내를
  반환한다(`confidence: LOW`, `citations: ["fallback-template ..."]`). 안전 시스템의 "무응답"을
  방지하기 위함. Backend는 항상 유효한 `response-guidance.v1`을 받는다고 가정할 수 있다.

## 9. 계약 변경 체크리스트 (한 PR에서 함께 수정)

- [ ] `agent-request.v1.json` / `response-guidance.v1.json` (Schema)
- [ ] 정상 예제 payload (`examples/`)
- [ ] agent `handler.py` 입력 파싱 / 출력 형식
- [ ] Backend 호출부 · `EventRecord` 저장 필드
- [ ] Frontend guidance 렌더링 타입
- [ ] 이 문서(표/예시)
- [ ] 호환 유지 or `schemaVersion` 증가
