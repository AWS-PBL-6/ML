"""
AE-Sentinel · RAG Response Guidance Layer
Response Guidance Lambda

역할:
  위험판정(NORMAL/CAUTION/DANGER) 결과를 입력으로 받아, Amazon Bedrock
  Knowledge Base(RAG)에서 계류삭 상황별 대처방안을 검색·생성하고 Backend에
  구조화 JSON(response-guidance.v1)을 반환한다.

설계 원칙(문서 준수):
  - 위험도는 이미 확정된 값이다. 재판정하지 않는다.
  - 스냅백 반경/폴리곤 등 물리 수치를 새로 계산하지 않는다(물리 엔진 몫).
  - 대처방안은 Knowledge Base 검색 근거에 기반한다(RAG).
  - 저장·WebSocket·SNS 이메일·SMS 발송은 Backend만 담당한다.

호출 방식:
  - ae-sentinel-app-ingest(오케스트레이터)에서 동기/비동기 Invoke, 또는
  - 콘솔 테스트 이벤트로 직접 실행.

Bedrock 호출 방식:
  - bedrock-agent-runtime.retrieve로 승인 문서를 검색한다.
  - bedrock-runtime.converse로 검색 근거와 판정 입력을 조합해 JSON을 생성한다.
"""

import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ---- 환경변수 ---------------------------------------------------------------
KNOWLEDGE_BASE_ID = os.environ.get("KNOWLEDGE_BASE_ID", "")
MODEL_ARN = os.environ.get("MODEL_ARN", "")
REGION = os.environ.get("AWS_REGION", "ap-northeast-2")
REQUIRE_APPROVED_KB = os.environ.get("REQUIRE_APPROVED_KB", "true").lower() in {"1", "true", "yes", "on"}

_bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", region_name=REGION)
_bedrock_runtime = boto3.client("bedrock-runtime", region_name=REGION)

VALID_RISK = {"NORMAL", "CAUTION", "DANGER"}


# ---- 입력 정규화 ------------------------------------------------------------
def _extract_decision(event: dict) -> dict:
    """
    다양한 호출 형태를 하나의 decision dict로 정규화한다.
    허용 입력:
      - 오케스트레이터 내부 payload({ "payload": {...} } 또는 평면 dict)
      - inference-response 필드(riskLevel/riskScore) + 컨텍스트(context)
    """
    body = event.get("payload", event) if isinstance(event, dict) else {}
    ctx = body.get("context", {}) or {}
    features = body.get("features", {}) or {}

    if body.get("schemaVersion") != "1.0.0":
        raise ValueError("schemaVersion must be '1.0.0'")
    missing = [key for key in ("traceId", "eventId", "siteId", "lineId") if not body.get(key)]
    if missing:
        raise ValueError(f"missing required field(s): {', '.join(missing)}")
    if not isinstance(features, dict) or not isinstance(ctx, dict):
        raise ValueError("features and context must be JSON objects")

    risk_level = str(body.get("riskLevel", "")).upper()
    if risk_level not in VALID_RISK:
        raise ValueError(f"riskLevel invalid or missing: {risk_level!r}")
    risk_score = body.get("riskScore")
    if isinstance(risk_score, bool) or not isinstance(risk_score, (int, float)) or not 0 <= risk_score <= 1:
        raise ValueError("riskScore must be a number between 0 and 1")

    return {
        "schemaVersion": body.get("schemaVersion"),
        "requestId": body.get("requestId"),
        "traceId": body.get("traceId") or f"trc-{uuid.uuid4().hex[:12]}",
        "eventId": body.get("eventId") or f"evt-{uuid.uuid4().hex[:12]}",
        "siteId": body.get("siteId", "unknown"),
        "berthId": body.get("berthId", "unknown"),
        "lineId": body.get("lineId", "unknown"),
        "bollardId": body.get("bollardId", "unknown"),
        "vesselId": body.get("vesselId", "미계선"),
        "riskLevel": risk_level,
        "riskScore": risk_score,
        "classProbabilities": body.get("classProbabilities", {}) or {},
        "modelVersion": body.get("modelVersion", "unknown"),
        "features": features,
        "context": ctx,
    }


def _build_query(d: dict) -> str:
    f = d["features"]
    c = d["context"]

    def g(src, key, default="정보 없음"):
        v = src.get(key)
        return default if v is None else v

    lines = [
        "[계류삭 위험 대응 요청]",
        f"사이트: {d['siteId']}",
        f"선석: {d['berthId']}",
        f"계류삭: {d['lineId']} (볼라드 {d['bollardId']}, 선박 {d['vesselId']})",
        f"확정 위험도: {d['riskLevel']} (riskScore={d['riskScore']}, model={d['modelVersion']})",
        "센서·판정 근거(실측/파생 구분):",
        f"- 실측 구조 진동수: {g(c, 'structuralFreqHz')} Hz (sensorKind={g(c, 'sensorKind')})",
        f"- 진동수 기반 파생 loadRatio: {g(c, 'loadRatio')}",
        f"- 파생 AE proxy: 진폭 {g(f, 'amplitudeDbAe')} dB, SNR {g(f, 'snrDb')} dB, "
        f"hitCount {g(f, 'hitCount')}, duration {g(f, 'durationMs')} ms",
        f"- 파생 주파수 대역: {g(f, 'freqLowKhz')}~{g(f, 'freqHighKhz')} kHz, "
        f"signalType={g(f, 'signalType')}",
        f"- 환경 컨텍스트: 소음 {g(c, 'ambientNoiseDb')} dB, 강우 {g(c, 'rainMmh')} mm/h, "
        f"풍속 {g(c, 'windMps')} m/s, 크레인 가동 {g(c, 'craneActive')}",
        "",
        "위 상황에서 안전관리자가 취해야 할 대처방안을 지식 기반에서 찾아 안내하라.",
        f"{d['siteId']}에 해당하는 항만사 전용 규정이 있으면 우선 적용하라.",
        "입력에 없는 로프 재질·실장력·MBL·물리 수치는 추정하거나 단정하지 마라.",
        "지정된 JSON 형식으로만 답하라.",
    ]
    return "\n".join(lines)


# ---- Bedrock 호출 -----------------------------------------------------------
# Lambda가 KB 검색 결과와 질의를 직접 치환해 모델에 보내는 프롬프트 템플릿.
_KB_PROMPT_TEMPLATE = """당신은 항만 계류삭(홋줄) 안전관제 대응 안내 도우미다.
아래 검색된 안전규정 발췌를 근거로, 관리자가 즉시 실행할 대처방안을 생성하라.
규칙: (1) 검색 결과에 없는 내용을 지어내지 마라. (2) 위험도는 확정값이므로 바꾸지 마라.
(3) 스냅백 반경·장력·MBL 등 물리 수치를 새로 계산하지 마라. (4) 항만사 전용 규정(port-specific)이
있으면 일반 지침보다 우선 적용하라. (5) NORMAL이면 immediateActions는 비워라.
(6) 검색 근거가 명시하지 않은 계류 해제, 선박 이동, 장력을 받는 줄의 조작을 지시하지 마라.
(7) DANGER에서는 인원 이탈 확인, 잠재적 반동 위험구역 접근 통제, 작업 중지, 안전 확보 후
자격 있는 담당자의 점검·보고를 우선하라.

검색된 규정:
$search_results$

요청:
$query$

반드시 아래 JSON 객체 하나만 출력하라. JSON 앞뒤에 설명 문장이나 마크다운 코드펜스(```)를
붙이지 마라. 모든 문장은 한국어로 작성하라.
{"riskLevel":"NORMAL|CAUTION|DANGER","headline":"관리자용 한 줄 요약(90자 이내)","situation":"현재 상황 2~3문장","immediateActions":["즉시 조치"],"followupActions":["후속 조치"],"citations":["참조한 규정 문서·항목"],"confidence":"HIGH|MEDIUM|LOW"}"""


def _reference_label(reference: dict) -> str | None:
    metadata = reference.get("metadata") or {}
    for key in ("title", "documentTitle", "source"):
        if metadata.get(key):
            return str(metadata[key])
    location = reference.get("location") or {}
    for value in location.values():
        if not isinstance(value, dict):
            continue
        for key in ("uri", "url"):
            if value.get(key):
                return str(value[key])
    return None


def _approved_site_filter(site_id: str) -> dict:
    # Bedrock metadata filters allow at most two nested logical-operator
    # levels. Express approval AND (general OR matching port-specific) as DNF:
    # (approval AND general) OR (approval AND port-specific AND site match).
    return {
        "orAll": [
            {
                "andAll": [
                    {"equals": {"key": "approvalStatus", "value": "approved"}},
                    {"equals": {"key": "scope", "value": "general"}},
                ]
            },
            {
                "andAll": [
                    {"equals": {"key": "approvalStatus", "value": "approved"}},
                    {"equals": {"key": "scope", "value": "port-specific"}},
                    {"equals": {"key": "siteId", "value": site_id}},
                ]
            },
        ]
    }


def _call_kb(query: str, decision: dict) -> tuple[str, list[str]]:
    """Retrieve approved KB passages, then generate once with those passages."""
    vector_config = {"numberOfResults": 6}
    if REQUIRE_APPROVED_KB:
        vector_config["filter"] = _approved_site_filter(decision["siteId"])
    risk_search = {
        "DANGER": (
            "계류삭 파단, 반동(snap-back), 장력을 받는 줄의 갑작스러운 움직임 위험에서 "
            "인원 이탈, 위험구역 접근 통제, 작업 중지, 안전 점검과 보고 절차"
        ),
        "CAUTION": (
            "계류삭 이상 징후와 마모·손상·과부하 예방을 위한 접근 통제, 검사, 감시 강화, "
            "안전한 유지보수 절차"
        ),
        "NORMAL": "계류삭 정상 상태의 정기 점검과 기록 유지",
    }[decision["riskLevel"]]
    retrieval_query = (
        f"한국 공식 계류 안전 문서에서 다음 관리자 대응 근거를 찾아라. "
        f"사이트 {decision['siteId']}, 확정 위험도 {decision['riskLevel']}: {risk_search}"
    )
    retrieval = _bedrock_agent_runtime.retrieve(
        knowledgeBaseId=KNOWLEDGE_BASE_ID,
        retrievalQuery={"text": retrieval_query},
        retrievalConfiguration={"vectorSearchConfiguration": vector_config},
    )
    passages: list[str] = []
    citations: list[str] = []
    for index, result in enumerate(retrieval.get("retrievalResults", []) or [], start=1):
        label = _reference_label(result) or f"검색 근거 {index}"
        if label not in citations:
            citations.append(label)
        content = result.get("content") or {}
        excerpt = str(content.get("text") or "").strip()
        if excerpt:
            passages.append(f"[근거 {index}: {label}]\n{excerpt[:3000]}")

    if not passages:
        return "", []

    prompt = _KB_PROMPT_TEMPLATE.replace("$search_results$", "\n\n".join(passages)).replace("$query$", query)
    generated = _bedrock_runtime.converse(
        modelId=MODEL_ARN,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 2048, "temperature": 0.0},
    )
    blocks = generated.get("output", {}).get("message", {}).get("content", []) or []
    text = "".join(str(block.get("text") or "") for block in blocks if isinstance(block, dict))
    return text, citations


# ---- 모델 출력 파싱 ---------------------------------------------------------
def _parse_guidance(raw_text: str, decision: dict, retrieved_citations: list[str]) -> dict:
    """모델이 반환한 텍스트에서 JSON 블록을 안전하게 추출한다."""
    parsed = None
    used_fallback = False
    if raw_text:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                logger.warning("guidance JSON parse failed; using fallback")

    if not isinstance(parsed, dict):
        parsed = _fallback_guidance(decision)
        used_fallback = True

    # 위험도는 확정값으로 강제(모델이 바꿔도 무시)
    parsed["riskLevel"] = decision["riskLevel"]
    parsed["headline"] = str(parsed.get("headline") or f"{decision['lineId']} {decision['riskLevel']}")[:90]
    parsed["situation"] = str(parsed.get("situation") or "")
    for key in ("immediateActions", "followupActions"):
        value = parsed.get(key)
        parsed[key] = [str(item) for item in value] if isinstance(value, list) else []
    # Bedrock retrieval attribution is authoritative; do not expose a model-
    # invented document title as if it were a real PDF/source.
    if retrieved_citations:
        parsed["citations"] = retrieved_citations
    elif not used_fallback:
        parsed["citations"] = []
        parsed["confidence"] = "LOW"
    if parsed.get("confidence") not in {"HIGH", "MEDIUM", "LOW"}:
        parsed["confidence"] = "LOW"
    return parsed


def _fallback_guidance(d: dict) -> dict:
    """Bedrock 호출/파싱 실패 시에도 최소한의 안내를 보장하는 안전망."""
    templates = {
        "NORMAL": {
            "headline": f"{d['lineId']} 정상. 특이사항 없음",
            "situation": "계류삭이 정상 범위입니다. 정기 순찰과 기록을 유지하십시오.",
            "immediate": [],
            "followup": ["정기 순찰 유지", "상태 기록 갱신"],
        },
        "CAUTION": {
            "headline": f"{d['lineId']} 주의. 육안 점검과 감시 강화 필요",
            "situation": "계류삭 장력 상승/이상 징후가 감지되었습니다. 예방 조치가 필요합니다.",
            "immediate": ["현장 육안 점검 지시(마모·소선 절단·과열)", "해당 라인 감시주기 단축"],
            "followup": ["장력 재분배 검토", "풍속·크레인 등 외력 요인 확인", "조치 내용 기록·보고"],
        },
        "DANGER": {
            "headline": f"{d['lineId']} 위험. 데드존 통제·하역 중단·인원 이탈 확인",
            "situation": "계류삭 파단 임박, 스냅백 위험 상태입니다. 인명 보호가 최우선입니다.",
            "immediate": ["스냅백 위험구역 내 인원 이탈 완료 확인", "해당 선석 하역 작업 즉시 중단", "위험구역 접근 통제"],
            "followup": ["선박·도선사에 계류 재조정 요청", "인접 라인 연쇄 파단 가능성 점검", "비상 보고 체계 가동"],
        },
    }
    t = templates[d["riskLevel"]]
    return {
        "headline": t["headline"],
        "situation": t["situation"],
        "immediateActions": t["immediate"],
        "followupActions": t["followup"],
        "citations": ["fallback-template (Bedrock 미응답)"],
        "confidence": "LOW",
    }


# ---- 엔트리포인트 -----------------------------------------------------------
def handler(event, context):
    logger.info("event=%s", json.dumps(event, ensure_ascii=False)[:2000])
    decision = _extract_decision(event)
    query = _build_query(decision)

    raw_text = ""
    citations = []
    try:
        raw_text, citations = _call_kb(query, decision)
    except Exception as exc:  # noqa: BLE001 - PoC 안전망: 실패해도 안내 보장
        logger.exception("Bedrock 호출 실패, fallback 사용: %s", exc)

    logger.info("bedrock raw_text=%s", (raw_text or "")[:1500])
    guidance = _parse_guidance(raw_text, decision, citations)

    # response-guidance.v1 계약에 맞춘 반환값
    result = {
        "schemaVersion": "1.0.0",
        "type": "response.guidance.generated",
        "traceId": decision["traceId"],
        "eventId": decision["eventId"],
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "lineId": decision["lineId"],
        "vesselId": decision["vesselId"],
        "siteId": decision["siteId"],
        "riskLevel": decision["riskLevel"],
        "riskScore": decision["riskScore"],
        "guidance": guidance,
        "source": "KB",
        "alertDispatched": False,
    }
    logger.info("result=%s", json.dumps(result, ensure_ascii=False)[:2000])
    return result
