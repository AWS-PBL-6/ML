# [팀 공유] Agent-Based Response Layer 구현 완료

> 담당 계층: 아키텍처의 **Agent-Based Response Layer**
> 상태: **AWS 배포 완료 · end-to-end 동작 확인**
> 계정: `727847798739` · 리전: `ap-northeast-2 (서울)`

---

## 1. 이 계층이 하는 일 (한 줄)

ML 위험판정(`NORMAL/CAUTION/DANGER`)을 받아, **Bedrock Knowledge Base(RAG)에서 상황별
대처방안을 검색·생성**하고 **관리자에게 알림(SNS) + 대시보드용 JSON**을 돌려준다.

```
[ML 위험판정] → (이 Lambda) → Bedrock KB(RAG, 규정집 검색)
                             → 관리자 이메일/문자 알림 (SNS)
                             → 대시보드용 구조화 JSON 반환
```

- 규정 출처: ① 일반 안전 매뉴얼(공통) ② 항만사별 개별 규정(`siteId`별, 항만사 규정 우선)
- 현장 작업자 대피 경보는 별도 계층 담당. **이 계층은 '관리자' 대상 후속 대처 안내.**

---

## 2. 배포된 AWS 리소스

| 리소스 | 이름/식별자 |
|---|---|
| Lambda | `ae-sentinel-response-guidance` |
| Bedrock Knowledge Base | `ae-sentinel-mooring-kb` (RAG, Titan Embeddings V2, OpenSearch Serverless) |
| 규정집 S3 버킷 | `ae-sentinel-kb-762616536500` (`general/`, `port-specific/<siteId>/`) |
| 생성 모델 | Claude 3.5 Sonnet (APAC 교차리전 추론 프로파일) |
| 알림 채널 | SNS 토픽 `ae-sentinel-alerts` |

---

## 3. 호출 방법 (ML/백엔드팀이 볼 부분)

위험판정 직후, 아래 형식의 JSON으로 이 Lambda를 **호출(비동기 Invoke 권장)** 하면 된다.
`riskLevel`이 `CAUTION` 또는 `DANGER`일 때만 알림이 나간다(`NORMAL`은 자동 억제).

### 입력(Input) 예시
```json
{
  "traceId": "trc-...",
  "eventId": "evt-...",
  "siteId": "yard-a",
  "lineId": "mooring-05",
  "bollardId": "bollard-05",
  "vesselId": "vessel-02",
  "riskLevel": "DANGER",
  "riskScore": 0.962,
  "features": { "rms": 0.82, "peakAmplitude": 0.96, "energy": 1450,
                "dominantFrequencyHz": 210000, "riseTimeMs": 3.2, "durationMs": 48 },
  "context": { "materialType": "synthetic-rope", "tensionEstimateKn": 168,
               "temperatureC": 26.1, "humidityPct": 78 }
}
```
- 필수: `riskLevel`(NORMAL/CAUTION/DANGER). 나머지는 있으면 안내 품질↑, 없으면 기본값 처리.
- **입력 계약**: `contracts/schemas/agent-request.v1.json` (예제: `contracts/examples/agent-request.v1.json`)
- **상세 인터페이스 문서**: `docs/backend-agent-interface.md`
- feature 어휘는 아키텍처 문서 6.2(`rms/peakAmplitude/energy/dominantFrequencyHz/riseTimeMs/durationMs`)를 따름
- 테스트 예시: `test/event-caution-line3.json`, `test/event-danger-line5.json`, `test/event-normal-line1.json`

### 출력(Output) = 대시보드/알림용
```json
{
  "schemaVersion": "1.0.0",
  "type": "response.guidance.generated",
  "traceId": "trc-...", "eventId": "evt-...",
  "lineId": "mooring-05", "siteId": "yard-a", "riskLevel": "DANGER",
  "guidance": {
    "headline": "관리자용 한 줄 요약(SMS용, 90자 이내)",
    "situation": "현재 상황 설명",
    "immediateActions": ["즉시 조치 ..."],
    "followupActions": ["후속 조치 ..."],
    "citations": ["참조한 규정 문서·항목"],
    "confidence": "HIGH|MEDIUM|LOW"
  },
  "source": "KB",
  "alertDispatched": true
}
```
- **정식 규격**: `contracts/schemas/response-guidance.v1.json` (예제: `contracts/examples/response-guidance.v1.json`)
- 대시보드팀은 이 `guidance` 객체를 그대로 화면에 렌더링하면 된다.

---

## 4. 동작 확인 완료

- ✅ **DANGER** → 규정집 근거(부산신항 전용 규정 우선) 대처방안 생성 + 관리자 이메일 발송
- ✅ **NORMAL** → 조치 없음 + 알림 억제
- ✅ Bedrock 장애 시에도 fallback 템플릿으로 안내 반환(무응답 방지)

---

## 5. 남은 통합 작업 (팀 협의 필요)

1. **ML → 이 Lambda 연결**: 위험판정 Lambda(오케스트레이터)가 위 입력 형식으로 이 Lambda를
   비동기 Invoke. (호출 측 역할에 `lambda:InvokeFunction` 권한 필요)
2. **대시보드 표시**: 반환 JSON을 이벤트 저장소에 적재 후 REST 조회, 또는 WebSocket로 요약 push.
   (실시간 push 시 `ws-risk-message.v1` 확장 여부는 계약 변경 프로세스로 협의)
3. **계정/리소스**: 현재는 훈련 계정(`727847798739`)에 배포. 통합 시 팀 공용 계정 기준으로 ARN 조정 필요.

---

## 6. 소스 위치

```
agent-response-layer/
  README.md                         # 전체 구현/콘솔 배포 가이드
  lambda/response_guidance/handler.py   # Lambda 코드
  knowledge-base/                   # 규정집(일반 + 항만사)
  agent/                            # 프롬프트/지시문
  iam/                              # 권한 정책
  test/                             # 테스트 입력 + 시나리오
  contracts/                        # 출력 규격(계약) 스키마 + 예제
```
