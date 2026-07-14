# RAG 지식문서 출처·승인 규칙

> 운영 원칙: 2026-07-14부터 S3 `approved/korean/`의 한국 공식 PDF만 운영 검색한다. 합성
> `.md`와 국제 PDF는 운영 S3에서 제거하고 검색 근거로 사용하지 않는다.

## 운영 승인 한국 문서

| 발행기관 | 문서 | 원출처 | 운영 용도 |
|---|---|---|---|
| 해양수산부 / 국가법령정보센터 | 선박설비기준 별표 21의2 | https://www.law.go.kr/LSW/admRulLsInfoP.do?admRulSeq=2100000271240 | 스냅백·동적 하중 노출 최소화, 과부하 방지 |
| 한국해양교통안전공단 | 해상교통안전진단·사전컨설팅 가이드(2023) | https://www.komsa.or.kr/thumbnail/dwld/kor/sub020302_file04.pdf | 계류삭 장력·안전성 평가 입력과 분석 항목 |
| 해양수산부 / 수협중앙회 | 연안선망어업 안전보건 표준매뉴얼 | https://www.suhyup.co.kr/bbs/suhyup/311/14647/download.do | 계류줄 파단 접근금지, 작업중지·대피 일반원칙 |

각 문서의 직접 적용 범위는 `approved/korean/*.metadata.json`의 `applicability`를 따른다.
KOMSA 2023 가이드는 평가 방법론, 수협 문서는 어업 분야 안전원칙으로만 사용한다. 특정 항만의
승인 절차가 등록되면 그 문서를 최우선으로 적용한다.

## 항만사 PDF 등록

항만사 규정은 공개 문서로 대신 만들지 않는다. 해당 항만사가 제공한 승인본 PDF를
`approved/port-specific/<siteId>/`에 올리고 같은 이름의 메타데이터 파일을 둔다.

```json
{
  "metadataAttributes": {
    "approvalStatus": "approved",
    "scope": "port-specific",
    "siteId": "yard-a",
    "title": "실제 승인 문서명",
    "revision": "2026-07",
    "approvedBy": "안전관리 책임자"
  }
}
```

한국 일반 공식 PDF는 `scope=general`, `siteId=ALL`로 등록한다. 개정본 교체 시 기존 문서를
`retired/`로 이동하고 KB Sync 후 검색 테스트를 수행한다.

## 인용값

모델이 작성한 임의 문서명이 아니라 Bedrock 응답의 `retrievedReferences.location` URI 또는
문서 메타데이터 `title`을 `guidance.citations`에 넣는다. 검색 근거가 없으면 Backend의
fallback 안내(`confidence=LOW`)를 반환한다.
