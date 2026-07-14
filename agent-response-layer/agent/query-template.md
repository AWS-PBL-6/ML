# RAG 검색 / 생성 프롬프트 템플릿

Lambda가 `Retrieve`로 Knowledge Base를 검색하고 `Converse`로 생성할 입력을 만드는 규칙이다.

## 1. 질의 문자열 조립 규칙

위험판정 이벤트에서 아래 값을 추출해 자연어 질의로 조립한다.

```
[계류삭 위험 대응 요청]
사이트: {siteId}
계류삭: {lineId} (볼라드 {bollardId}, 선박 {vesselId})
위험도: {riskLevel} (riskScore={riskScore})
판정 근거 수치:
- 실측 구조 진동수: {structuralFreqHz} Hz
- 진동수 기반 파생 loadRatio: {loadRatio}
- 파생 AE proxy: 진폭 {amplitudeDbAe} dB, SNR {snrDb} dB, hitCount {hitCount}
- 파생 주파수 대역: {freqLowKhz}~{freqHighKhz} kHz, signalType {signalType}
- 배경소음 {ambientNoiseDb} dB, 강우 {rainMmh} mm/h, 풍속 {windMps} m/s, 크레인 가동 {craneActive}

위 상황에서 안전관리자가 취해야 할 대처방안을 지식 기반에서 찾아 안내하라.
{siteId}에 해당하는 항만사 전용 규정이 있으면 우선 적용하라.
지정된 JSON 형식으로만 답하라.
```

> 피에조 원본 사실값은 `structuralFreqHz`이며 loadRatio와 AE 특징은 Backend 파생값이다.
> 입력에 없는 재질·장력·MBL은 채우거나 추정하지 않는다.

## 2. 검색 결과 + 모델 생성 프롬프트

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
