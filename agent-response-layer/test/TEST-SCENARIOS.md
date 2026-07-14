# 테스트 시나리오 · 검증 절차

모든 명령은 현재 Application 계정, 리전 `ap-northeast-2`, 프로젝트용 AWS 프로필 기준이다.
프로필이 다르면 `--profile` 값을 바꾼다.

## 0. 준비물

- 배포된 Lambda 이름: `ae-sentinel-response-guidance`
- 생성된 Knowledge Base ID: `<KB_ID>`
- 알림 검증은 Response Guidance Lambda 단독 테스트가 아니라 Backend 목업 경보 API에서 수행한다.

---

## 1. Knowledge Base 단독 검증 (RAG가 근거를 찾는가)

Lambda 없이 KB만 먼저 확인한다. 콘솔: Bedrock → Knowledge Bases → 대상 KB → **Test** 패널에서
아래 질문을 입력.

- 질문 예: `yard-a 사이트에서 5번 계류삭이 DANGER일 때 관리자가 할 조치는?`
- 기대: 부산신항 A터미널 규정 §4와 일반지침 §3 내용이 인용되어 답변에 반영.

CLI로도 가능:

```bash
aws bedrock-agent-runtime retrieve-and-generate --region ap-northeast-2 --profile ae-sentinel \
  --input '{"text":"yard-a 5번 계류삭 DANGER 상황 관리자 대처방안"}' \
  --retrieve-and-generate-configuration '{
    "type":"KNOWLEDGE_BASE",
    "knowledgeBaseConfiguration":{
      "knowledgeBaseId":"<KB_ID>",
      "modelArn":"<CURRENT_ACCOUNT_INFERENCE_PROFILE_ARN>"
    }
  }'
```

기대: `citations`에 general/port-specific 문서 청크가 포함되고, 답변이 데드존 통제·하역 중단을 언급.

---

## 2. Lambda 직접 호출 (end-to-end)

### 2-1. CAUTION (3번 라인)

```bash
aws lambda invoke --region ap-northeast-2 --profile ae-sentinel \
  --function-name ae-sentinel-response-guidance \
  --payload fileb://event-caution-line3.json \
  out-caution.json
type out-caution.json
```

기대(`out-caution.json`):
- `riskLevel` = `CAUTION`
- `guidance.immediateActions`에 "육안 점검", "감시주기 단축" 류 항목
- `guidance.citations`에 항만사 규정(§3) 또는 일반지침(§2) 근거
- `alertDispatched` = `true` (CAUTION은 NOTIFY_ON 기본 포함)

### 2-2. DANGER (5번 라인)

```bash
aws lambda invoke --region ap-northeast-2 --profile ae-sentinel \
  --function-name ae-sentinel-response-guidance \
  --payload fileb://event-danger-line5.json \
  out-danger.json
type out-danger.json
```

기대:
- `riskLevel` = `DANGER`
- `guidance.immediateActions`에 "데드존/위험구역 통제", "하역 중단", "인원 이탈 확인"
- `guidance.headline` 90자 이내
- `alertDispatched` = `true`

### 2-3. NORMAL (1번 라인) — 발송 억제 확인

```bash
aws lambda invoke --region ap-northeast-2 --profile ae-sentinel \
  --function-name ae-sentinel-response-guidance \
  --payload fileb://event-normal-line1.json \
  out-normal.json
type out-normal.json
```

기대:
- `riskLevel` = `NORMAL`
- `guidance.immediateActions` = `[]` (즉시 조치 없음)
- `alertDispatched` = `false` (NORMAL은 NOTIFY_ON 미포함 → SNS 미발송)

> Windows CMD에서는 `type`, PowerShell/bash에서는 `cat`으로 결과 파일을 연다.

---

## 3. SNS 수신 확인

- 이메일 구독: 받은편지함에서 `[AE-Sentinel] DANGER mooring-05` 제목 메일 확인.
- SMS 구독: 문자 수신 확인(헤드라인 + 즉시 조치 요약).
- 2-3(NORMAL)에서는 알림이 오지 않아야 한다.

---

## 4. 검증 체크리스트

| # | 항목 | 통과 기준 |
|---|---|---|
| 1 | KB 검색 근거 | 답변에 general + port-specific 인용이 모두 나타남 |
| 2 | 위험도 고정 | 입력 riskLevel이 출력에서 절대 바뀌지 않음 |
| 3 | 물리 미계산 | 반경/폴리곤 등 새 수치를 지어내지 않음 |
| 4 | CAUTION 안내 | 예방·감시 중심 조치 |
| 5 | DANGER 안내 | 인명보호·통제·하역중단 중심 조치 |
| 6 | NORMAL 억제 | SNS 미발송, immediateActions 비어 있음 |
| 7 | 사이트 우선 | siteId=yard-a일 때 항만사 규정 우선 반영 |
| 8 | 안전망 | Bedrock 실패 시 fallback 템플릿으로 안내 반환(500 아님) |
| 9 | 계약 정합 | 출력이 response-guidance.v1.json 스키마를 만족 |

---

## 5. 실패 시 점검 순서

1. `AccessDeniedException` → Lambda 실행역할에 `bedrock:RetrieveAndGenerate`/`bedrock:InvokeModel` 및
   해당 KB ARN이 있는지 확인. 모델 액세스(Bedrock 콘솔 → Model access)가 활성인지 확인.
2. `ResourceNotFoundException` → `KNOWLEDGE_BASE_ID` / `MODEL_ARN` 오타 확인, 리전 일치 확인.
3. `ValidationException` (modelArn) → 서울 리전은 온디맨드 대신 **inference profile ARN**
   (`apac.anthropic...`)을 요구할 수 있음. profile ARN으로 교체.
4. SNS 미수신 → 구독 상태 `Confirmed`인지, `NOTIFY_ON`에 해당 위험도가 있는지 확인.
5. 근거 인용이 비어 있음 → KB 데이터 소스 **Sync** 완료 여부, S3 prefix/문서 업로드 확인.
