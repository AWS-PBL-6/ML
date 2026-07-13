# RAG 쿼리 / 프롬프트 템플릿

Lambda가 Bedrock에 보낼 입력 텍스트(질의)를 만드는 규칙이다. Agent 방식(`InvokeAgent`)과
Knowledge Base 직접 방식(`RetrieveAndGenerate`) 모두 동일한 질의 문자열을 사용한다.

## 1. 질의 문자열 조립 규칙

위험판정 이벤트에서 아래 값을 추출해 자연어 질의로 조립한다.

```
[계류삭 위험 대응 요청]
사이트: {siteId}
계류삭: {lineId} (볼라드 {bollardId}, 선박 {vesselId})
위험도: {riskLevel} (riskScore={riskScore})
판정 근거 수치:
- 접촉식 AE 진폭: {amplitudeDbAe} dB
- SNR: {snrDb} dB
- 히트카운트(합/최대): {hitSum} / {hitMax}
- 고주파 이벤트 수: {nHigh}
- 로프 재질: {materialType}, 추정 장력: {tensionEstimateKn} kN
- 온도 {temperatureC}C, 습도 {humidityPct}%, 풍속 {windMps} m/s, 크레인 가동 {craneActive}

위 상황에서 안전관리자가 취해야 할 대처방안을 지식 기반에서 찾아 안내하라.
{siteId}에 해당하는 항만사 전용 규정이 있으면 우선 적용하라.
지정된 JSON 형식으로만 답하라.
```

> 값이 없는 필드는 "정보 없음"으로 채우거나 해당 줄을 생략한다. Lambda가 안전하게 처리한다.

## 2. RetrieveAndGenerate 방식 promptTemplate (Agent 없이 KB만 쓸 때)

`RetrieveAndGenerate` API의 `generationConfiguration.promptTemplate.textPromptTemplate`에
아래를 넣는다. `$search_results$`, `$query$`는 Bedrock이 치환하는 예약 변수다.

```text
당신은 항만 계류삭 안전관제 대응 안내 도우미다. 아래 검색된 안전규정 발췌를 근거로,
관리자가 즉시 실행할 대처방안을 생성하라. 검색 결과에 없는 내용은 지어내지 마라.
위험도는 확정값이므로 바꾸지 마라. 물리 수치를 새로 계산하지 마라.
항만사 전용 규정(port-specific)이 있으면 일반 지침보다 우선 적용하라.

<검색된 규정>
$search_results$
</검색된 규정>

요청:
$query$

반드시 아래 JSON 하나만 출력하라(앞뒤 설명 금지):
{"riskLevel":"...","headline":"...","situation":"...","immediateActions":[...],"followupActions":[...],"citations":[...],"confidence":"..."}
```

## 3. 검색 필터(권장)

- `scope=general`은 항상 검색 대상.
- `scope=port-specific`은 `siteId` 메타데이터가 일치할 때 우선.
- Knowledge Base 메타데이터 필터를 쓰면 `retrievalConfiguration.vectorSearchConfiguration.filter`에
  `{"equals":{"key":"siteId","value":"yard-a"}}` 형태로 사이트 한정 검색이 가능하다.
  (필터 사용 시 general 문서에도 `siteId=ALL` 같은 공통 값을 부여하거나, general/port를
  각각 조회해 합치는 전략을 README의 튜닝 절에서 설명한다.)
