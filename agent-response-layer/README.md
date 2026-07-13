# Agent-Based Response Layer — 구현 가이드

> AE-Sentinel 계류삭 안전관제 시스템의 **에이전트 기반 대응 계층**.
> 위험판정(NORMAL/CAUTION/DANGER) 이후, Amazon Bedrock Knowledge Base(RAG)에서
> 상황별 대처방안을 검색·생성해 **관리자에게 SNS/대시보드로 전달**한다.
>
> - 계정: **762616536500** · 리전: **ap-northeast-2 (서울)**
> - 기존 스택 연동 대상: `ae-sentinel-app` (SAM) — [애플리케이션 계정 가이드](https://github.com/AWS-PBL-6/docs/blob/main/docs/operations/aws-application-account-guide.md)
> - 담당 계층: 아키텍처 다이어그램의 **Agent-Based Response Layer**

---

## 1. 이 계층이 하는 일

```
ML 위험판정(riskLevel)            Agent-Based Response Layer (이 문서)
  Risk Decision Lambda  ─────▶  Response Guidance Lambda
                                   │
                                   ├─▶ Bedrock Knowledge Base (RAG)
                                   │     └─ S3: 일반 안전매뉴얼 + 항만사 개별규정
                                   │
                                   ├─▶ SNS  ─▶ 관리자 휴대폰(SMS)/이메일
                                   └─▶ 반환 JSON ─▶ 관리자 대시보드
```

- 입력: 특정 계류삭의 위험도 + 판정 근거 수치 + 계류삭/볼라드/선박/사이트 식별자.
- 처리: Knowledge Base에서 해당 상황·해당 항만사에 맞는 대처방안을 RAG로 검색·생성.
- 출력: 관리자용 구조화 안내(`response-guidance.v1`) + SNS 통보.

### 설계 원칙 (문서 준수)
- **위험도는 확정값** — Bedrock은 재판정하지 않는다.
- **물리 미계산** — 스냅백 반경/폴리곤은 물리 엔진 몫. 여기선 설명·전달만.
- **RAG 근거 기반** — 대처방안은 KB 검색 결과에 근거하며 지어내지 않는다.
- **현장 대피 경보는 별도** — 이 계층은 *관리자* 대상 후속 대처 안내다.
- **항만사 규정 우선** — `siteId` 일치 규정을 일반 지침보다 우선 적용.

### 지식 출처 2종
1. `knowledge-base/general/` — 국가·공공 오픈소스 성격의 일반 안전관리 매뉴얼(공통).
2. `knowledge-base/port-specific/` — 항만사가 개별 등록하는 사내 규정집(서비스 이용회사가 넣음).

---

## 2. 폴더 구성

```text
agent-response-layer/
  README.md                      # 이 문서(마스터 가이드)
  knowledge-base/
    general/                     # S3 general/ prefix 로 업로드
    port-specific/               # S3 port-specific/yard-a/ prefix 로 업로드
  agent/
    agent-instruction.md         # Bedrock Agent 지시문(붙여넣기용)
    query-template.md            # RAG 질의/프롬프트 템플릿
  lambda/response_guidance/
    handler.py                   # Response Guidance Lambda (진입점 handler.handler)
    requirements.txt
  iam/
    lambda-trust-policy.json
    lambda-execution-policy.json
    knowledge-base-role-trust-policy.json
    knowledge-base-role-permissions-policy.json
  test/
    event-caution-line3.json / event-danger-line5.json / event-normal-line1.json
    TEST-SCENARIOS.md
  contracts/
    schemas/response-guidance.v1.json    # 출력 규격(계약)
    examples/response-guidance.v1.json
```

계약(이 저장소 내): `contracts/schemas/response-guidance.v1.json` (+ `contracts/examples/response-guidance.v1.json`).

---

## 3. 두 가지 구현 경로

| 경로 | 방식 | 장점 | 권장 |
|---|---|---|---|
| **A. KB 직접(RetrieveAndGenerate)** | Lambda가 `retrieve_and_generate` 호출 | 단순·저렴·구성 적음 | PoC 기본 |
| **B. Bedrock Agent(InvokeAgent)** | KB 연결한 Agent를 Lambda가 호출 | 다이어그램 일치, 확장(멀티턴/도구) 용이 | 발표 시연 |

Lambda는 환경변수 `GUIDANCE_MODE`(`KB`/`AGENT`)로 두 경로를 전환한다. 아래 4~9절은 공통이며,
Agent가 필요한 부분(7절)만 경로 B에서 추가 수행한다.

---

## 4. STEP 1 — 모델 사용 가능 상태 확인

> ⚠️ **정책 변경(2025)**: 기존 **Model access** 페이지는 **폐지**되었다. 이제 서버리스
> 파운데이션 모델은 계정에서 **처음 호출되는 순간 자동으로 활성화**된다. 수동으로 켜는
> 절차가 없어졌다. 단, **Anthropic(Claude) 모델은 첫 사용 시 "use case 양식" 제출**이
> 필요할 수 있고, AWS Marketplace 제공 모델은 권한 있는 사용자가 1회 호출해야 계정 전체에
> 활성화된다.

콘솔 → **Amazon Bedrock** → 좌측 **Model catalog** (또는 **Playground → Chat**).

1. 리전이 **ap-northeast-2 (서울)** 인지 우상단에서 확인.
2. **Claude 3.5 Sonnet** 을 찾아 **Open in Playground** → 아무 문장(`안녕`)을 입력하고 실행.
   - 정상 응답 → 이미 사용 가능. 완료.
   - **"submit use case details"** 화면이 뜨면 → 회사명·용도 등을 간단히 제출(보통 즉시~수 분
     내 승인) → 다시 실행해 응답 확인.
3. **Amazon Titan Text Embeddings V2** 는 별도 조치 불필요. STEP 3에서 Knowledge Base가
   처음 사용할 때 자동 활성화된다.

> 서울 리전에서 Claude는 **교차 리전 추론 프로파일**(`apac.anthropic....`)로 호출해야 할 수
> 있다. 온디맨드 모델 ARN으로 `ValidationException`이 나면 프로파일 ARN으로 교체한다
> (Bedrock → Cross-region inference 에서 프로파일 ID 확인).
>
> 참고: `bedrock:InvokeModel` 등 권한은 IAM/SCP로 관리된다. Lambda 실행역할에는 이미
> 해당 권한을 `iam/lambda-execution-policy.json` 에 넣어두었다(STEP 5~6).

---

## 5. STEP 2 — S3 버킷 만들고 지식 문서 업로드

콘솔 → **S3** → **Create bucket**.

1. 이름 예: `ae-sentinel-kb-762616536500` (전역 고유해야 함), 리전 서울.
2. 기본 보안 유지: **Block all public access = ON**, 기본 암호화(SSE-S3/KMS) 사용.
3. 버킷 안에 prefix 2개를 만들고 이 저장소 문서를 업로드:
   - `general/` ← `knowledge-base/general/*.md`
   - `port-specific/yard-a/` ← `knowledge-base/port-specific/*.md`

> 항만사별 규정은 사이트마다 `port-specific/<siteId>/` 로 분리해 넣는다. 새 항만사 온보딩 =
> 그 prefix에 문서 업로드 후 KB Sync만 하면 된다(코드 변경 없음).

### (선택) 메타데이터 필터용 사이드카
사이트 한정 검색을 쓰려면 각 문서 옆에 `<파일명>.metadata.json` 을 둔다. 예:
`port-specific/yard-a/busan-newport-mooring-regulation.md.metadata.json`
```json
{ "metadataAttributes": { "scope": "port-specific", "siteId": "yard-a" } }
```
general 문서에는 `{ "metadataAttributes": { "scope": "general", "siteId": "ALL" } }`.

---

## 6. STEP 3 — Bedrock Knowledge Base 생성

콘솔 → **Amazon Bedrock** → **Knowledge Bases** → **Create knowledge base**.

1. **이름**: `ae-sentinel-mooring-kb`.
2. **IAM 권한**: "Create and use a new service role" 선택(콘솔이 자동 생성). 직접 만들려면
   `iam/knowledge-base-role-*.json` 두 정책을 사용.
3. **데이터 소스**: Amazon S3 → 위 버킷 지정.
   - 데이터 소스 1: `s3://ae-sentinel-kb-.../general/`
   - 데이터 소스 2: `s3://ae-sentinel-kb-.../port-specific/` (한 KB에 소스 2개 추가 가능)
   - 청킹: 기본(Default chunking) 또는 문서가 짧으니 "Fixed size 300 tokens, 20% overlap".
4. **임베딩 모델**: **Titan Text Embeddings V2**.
5. **벡터 스토어**: "Quick create a new vector store" → **Amazon OpenSearch Serverless**
   (콘솔이 컬렉션까지 자동 생성. 가장 간단).
6. 생성 완료 후 **데이터 소스별 Sync** 실행. 상태가 **Available/Completed** 인지 확인.
7. 생성된 **Knowledge base ID** 를 기록(예: `ABCDEFGHIJ`). Lambda 환경변수에 쓴다.

> ⚠️ OpenSearch Serverless는 유휴에도 최소 과금(OCU)이 발생한다. PoC 종료 시 KB와 컬렉션을
> 반드시 삭제한다(11절). 비용을 아끼려면 벡터 스토어로 **Aurora Serverless v2(pgvector)** 나
> **Pinecone**도 선택 가능하나, 콘솔 자동생성은 OpenSearch Serverless가 가장 쉽다.

8. **동작 확인**: KB 상세 → 우측 **Test** 패널에서 모델 선택 후 질문 입력
   (`test/TEST-SCENARIOS.md` §1 참고). 인용(citations)이 나오면 정상.

---

## 7. STEP 4 (경로 B 전용) — Bedrock Agent 생성

경로 A(KB 직접)만 쓸 거면 이 절은 건너뛴다.

콘솔 → **Amazon Bedrock** → **Agents** → **Create Agent**.

1. **이름**: `ae-sentinel-response-agent`.
2. **모델**: Claude 3.5 Sonnet(또는 Haiku) — 서울은 프로파일 사용 가능.
3. **Instructions**: `agent/agent-instruction.md` 의 "Instruction 본문"을 그대로 붙여넣기.
4. **Knowledge Base 연결**: 6절에서 만든 `ae-sentinel-mooring-kb` 추가. KB description은
   `agent/agent-instruction.md` 하단 문구 사용.
5. **Action groups**: 없음(순수 RAG+생성).
6. **Prepare** → **Create Alias**(예: `prod`). **Agent ID** 와 **Alias ID** 기록.

---

## 8. STEP 5 — Response Guidance Lambda 배포

콘솔 → **Lambda** → **Create function** → Author from scratch.

1. **이름**: `ae-sentinel-response-guidance`, 런타임 **Python 3.12**, 아키텍처 arm64/x86 무관.
2. **실행 역할**: 새 역할 생성 후, 아래 정책을 붙인다(또는 미리 IAM에서 역할 생성).
   - 신뢰정책: `iam/lambda-trust-policy.json`
   - 권한정책: `iam/lambda-execution-policy.json` (`REPLACE_KB_ID`, `REPLACE_AGENT_ID`를 실제 값으로)
3. **코드**: `lambda/response_guidance/handler.py` 내용을 인라인 편집기에 붙여넣거나 zip 업로드.
   - **Handler** 설정: `handler.handler`
4. **Configuration → General**: Timeout **30초**, Memory **256MB** 권장(Bedrock 지연 대비).
5. **Configuration → Environment variables**:

| Key | 경로 A(KB) 값 | 경로 B(Agent) 값 |
|---|---|---|
| `GUIDANCE_MODE` | `KB` | `AGENT` |
| `KNOWLEDGE_BASE_ID` | `<KB_ID>` | (불필요, Agent가 참조) |
| `MODEL_ARN` | `arn:aws:bedrock:ap-northeast-2:762616536500:inference-profile/apac.anthropic.claude-3-5-sonnet-20240620-v1:0` | (불필요) |
| `AGENT_ID` | — | `<AGENT_ID>` |
| `AGENT_ALIAS_ID` | — | `<ALIAS_ID>` |
| `SNS_TOPIC_ARN` | `arn:aws:sns:ap-northeast-2:762616536500:ae-sentinel-app-alerts` | 동일 |
| `NOTIFY_ON` | `CAUTION,DANGER` | 동일 |

> `MODEL_ARN`은 온디맨드(`arn:aws:bedrock:ap-northeast-2::foundation-model/anthropic.claude-3-5-sonnet-20240620-v1:0`)로
> 먼저 시도하고, `ValidationException`이 나면 위의 inference-profile ARN으로 교체.

---

## 9. STEP 6 — SNS 통보 대상 연결

기존 `ae-sentinel-app-alerts` 토픽을 재사용한다(별도 생성 불필요).

콘솔 → **SNS** → Topics → `ae-sentinel-app-alerts` → **Create subscription**.
- 이메일: Protocol=Email, Endpoint=관리자 메일 → 수신 메일에서 **Confirm** 클릭.
- SMS: Protocol=SMS, Endpoint=관리자 휴대폰(E.164, 예: `+8210...`).

> SMS는 리전/계정의 SMS 샌드박스·발신자 등록 정책의 영향을 받을 수 있다. 데모는 이메일
> 구독이 가장 확실하다. Lambda 실행역할에는 이미 `sns:Publish` 권한이 있다.

---

## 10. STEP 7 — 위험판정 흐름과 연결(오케스트레이터)

이 계층은 위험판정 직후 호출되어야 한다. 두 가지 연결 방법:

### 방법 1 — 오케스트레이터에서 직접 Invoke (권장, 지연 낮음)
`ae-sentinel-app-ingest`(오케스트레이터)가 추론 후 위험도가 CAUTION/DANGER면
`ae-sentinel-response-guidance`를 **비동기(Event) Invoke** 한다.
- ingest 실행역할에 `lambda:InvokeFunction`(대상 함수 ARN) 권한 추가.
- 전달 payload는 `test/event-*.json`과 동일한 형태(평면 dict + `features` + `context`).
- SAM으로 관리하므로 `backend/infra/application-account/template.yaml`에 함수/환경변수/권한을
  추가하고 `sam deploy` (담당: 김태훈과 협의).

### 방법 2 — EventBridge/DynamoDB Streams 디커플링 (확장형)
ingest가 `-events` 테이블에 위험 이벤트를 쓰면, DynamoDB Streams 또는 EventBridge 규칙이
`ae-sentinel-response-guidance`를 트리거. 실시간 경로와 분리돼 장애 격리에 유리.

### 대시보드 전달
- Lambda 반환값(`response-guidance.v1`)을 오케스트레이터가 받아 `-events`/`-lines`에 저장하면,
  대시보드가 REST(`GET /v1/events/{eventId}`)로 조회.
- 즉시 표시가 필요하면 WebSocket push에 요약을 실어 보낸다. 단, `ws-risk-message.v1`은
  `additionalProperties:false`라 **구조화 guidance를 그대로 넣으려면 계약 확장이 필요**하다.
  단기적으로는 `alertMessage`에 `guidance.headline`을 넣고, 상세는 REST 조회로 푼다.

---

## 11. 비용 · 정리

- **과금 요소**: Bedrock 모델 호출(토큰당), Titan 임베딩(문서 인덱싱 1회+쿼리), Lambda/SNS(종량),
  **OpenSearch Serverless(유휴에도 최소 OCU 과금)**.
- **PoC 종료 시 정리 순서**: Agent 삭제 → Knowledge Base 삭제 → OpenSearch Serverless 컬렉션 삭제
  → Lambda 삭제 → (필요 시) S3 버킷 비우고 삭제. `ae-sentinel-app-alerts` 토픽은 App 스택
  소유이므로 임의 삭제 금지.
- 상시 과금이 걱정되면 벡터 스토어를 Aurora Serverless v2로 바꾸거나, 시연 직전에 KB를
  생성하고 직후 삭제하는 방식을 권장.

---

## 12. 계약(Contract) 주의

- 산출물 계약 `contracts/schemas/response-guidance.v1.json`은 **기존 스키마를
  건드리지 않는 additive 추가**다. 팀 공용 계약이므로 `docs` 저장소의 계약 변경 프로세스
  (`contracts-change-process`)에 따라 **팀 합의 후 확정**한다.
- 필드/enum 변경 시 `schemaVersion`을 올리고 examples를 함께 갱신한다.

---

## 13. 보안 메모

- KB 소스 S3 버킷은 **퍼블릭 접근 차단** 유지, 서버측 암호화 사용.
- Lambda 실행역할·KB 서비스역할은 이 문서의 IAM 정책처럼 **최소 권한**(특정 KB/모델/토픽 ARN)만 부여.
- KB 서비스역할 신뢰정책에 `aws:SourceAccount`/`aws:SourceArn` 조건을 걸어 confused-deputy를 방지.
- 관리자 알림 채널(SNS)은 인증된 구독만 두고, 개인정보(전화번호)는 로그에 남기지 않는다
  (핸들러는 전화번호를 로깅하지 않는다).
- 항만사 규정 문서에 민감정보가 있으면 KMS 고객관리키(CMK)와 버킷 정책으로 접근을 좁힌다.
