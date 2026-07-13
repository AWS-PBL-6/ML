"""
AE-Sentinel · Agent-Based Response Layer
Response Guidance Lambda

역할:
  위험판정(NORMAL/CAUTION/DANGER) 결과를 입력으로 받아, Amazon Bedrock
  Knowledge Base(RAG)에서 계류삭 상황별 대처방안을 검색·생성하고, 관리자에게
  SNS로 통보한 뒤 대시보드용 구조화 JSON(response-guidance.v1)을 반환한다.

설계 원칙(문서 준수):
  - 위험도는 이미 확정된 값이다. 재판정하지 않는다.
  - 스냅백 반경/폴리곤 등 물리 수치를 새로 계산하지 않는다(물리 엔진 몫).
  - 대처방안은 Knowledge Base 검색 근거에 기반한다(RAG).
  - 현장 작업자 대피 명령은 별도 경보로 이미 발령됨. 여기서는 '관리자' 안내만 담당.

호출 방식:
  - ae-sentinel-app-ingest(오케스트레이터)에서 동기/비동기 Invoke, 또는
  - 콘솔 테스트 이벤트로 직접 실행.

Bedrock 호출 모드(환경변수 GUIDANCE_MODE):
  - "KB"    : bedrock-agent-runtime.retrieve_and_generate (기본, 단순/저렴)
  - "AGENT" : bedrock-agent-runtime.invoke_agent (다이어그램의 Bedrock Agent)
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
GUIDANCE_MODE = os.environ.get("GUIDANCE_MODE", "KB").upper()   # KB | AGENT
KNOWLEDGE_BASE_ID = os.environ.get("KNOWLEDGE_BASE_ID", "")
MODEL_ARN = os.environ.get("MODEL_ARN", "")                     # KB 모드용 모델/추론프로파일 ARN
AGENT_ID = os.environ.get("AGENT_ID", "")                       # AGENT 모드용
AGENT_ALIAS_ID = os.environ.get("AGENT_ALIAS_ID", "")           # AGENT 모드용
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "")
NOTIFY_ON = {s.strip().upper() for s in os.environ.get("NOTIFY_ON", "CAUTION,DANGER").split(",") if s.strip()}
REGION = os.environ.get("AWS_REGION", "ap-northeast-2")

_bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", region_name=REGION)
_sns = boto3.client("sns", region_name=REGION)

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

    risk_level = str(body.get("riskLevel", "")).upper()
    if risk_level not in VALID_RISK:
        raise ValueError(f"riskLevel invalid or missing: {risk_level!r}")

    return {
        "traceId": body.get("traceId") or f"trc-{uuid.uuid4().hex[:12]}",
        "eventId": body.get("eventId") or f"evt-{uuid.uuid4().hex[:12]}",
        "siteId": body.get("siteId", "unknown"),
        "lineId": body.get("lineId", "unknown"),
        "bollardId": body.get("bollardId", "unknown"),
        "vesselId": body.get("vesselId", "미계선"),
        "riskLevel": risk_level,
        "riskScore": body.get("riskScore", 0),
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
        f"계류삭: {d['lineId']} (볼라드 {d['bollardId']}, 선박 {d['vesselId']})",
        f"위험도: {d['riskLevel']} (riskScore={d['riskScore']})",
        "판정 근거 수치:",
        f"- 접촉식 AE 진폭: {g(f, 'amplitudeDbAe')} dB",
        f"- SNR: {g(f, 'snrDb')} dB",
        f"- 히트카운트(합/최대): {g(f, 'hitSum')} / {g(f, 'hitMax')}",
        f"- 고주파 이벤트 수: {g(f, 'nHigh')}",
        f"- 로프 재질: {g(c, 'materialType')}, 추정 장력: {g(c, 'tensionEstimateKn')} kN",
        f"- 온도 {g(c, 'temperatureC')}C, 습도 {g(c, 'humidityPct')}%, "
        f"풍속 {g(c, 'windMps')} m/s, 크레인 가동 {g(c, 'craneActive')}",
        "",
        "위 상황에서 안전관리자가 취해야 할 대처방안을 지식 기반에서 찾아 안내하라.",
        f"{d['siteId']}에 해당하는 항만사 전용 규정이 있으면 우선 적용하라.",
        "지정된 JSON 형식으로만 답하라.",
    ]
    return "\n".join(lines)


# ---- Bedrock 호출 -----------------------------------------------------------
# RetrieveAndGenerate 전용 프롬프트 템플릿.
# $search_results$ = KB에서 검색된 규정 발췌(필수 변수), $query$ = 우리가 만든 질의.
# 이 템플릿이 없으면 Bedrock 기본 프롬프트가 자연어로 답해서 JSON 파싱이 안 된다.
_KB_PROMPT_TEMPLATE = """당신은 항만 계류삭(홋줄) 안전관제 대응 안내 도우미다.
아래 검색된 안전규정 발췌를 근거로, 관리자가 즉시 실행할 대처방안을 생성하라.
규칙: (1) 검색 결과에 없는 내용을 지어내지 마라. (2) 위험도는 확정값이므로 바꾸지 마라.
(3) 스냅백 반경 등 물리 수치를 새로 계산하지 마라. (4) 항만사 전용 규정(port-specific)이
있으면 일반 지침보다 우선 적용하라. (5) NORMAL이면 immediateActions는 비워라.

검색된 규정:
$search_results$

요청:
$query$

반드시 아래 JSON 객체 하나만 출력하라. JSON 앞뒤에 설명 문장이나 마크다운 코드펜스(```)를
붙이지 마라. 모든 문장은 한국어로 작성하라.
{"riskLevel":"NORMAL|CAUTION|DANGER","headline":"관리자용 한 줄 요약(90자 이내)","situation":"현재 상황 2~3문장","immediateActions":["즉시 조치"],"followupActions":["후속 조치"],"citations":["참조한 규정 문서·항목"],"confidence":"HIGH|MEDIUM|LOW"}"""


def _call_kb(query: str) -> str:
    """RetrieveAndGenerate: KB 검색 + 생성 한 번에."""
    resp = _bedrock_agent_runtime.retrieve_and_generate(
        input={"text": query},
        retrieveAndGenerateConfiguration={
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": KNOWLEDGE_BASE_ID,
                "modelArn": MODEL_ARN,
                "retrievalConfiguration": {
                    "vectorSearchConfiguration": {"numberOfResults": 6}
                },
                "generationConfiguration": {
                    "promptTemplate": {"textPromptTemplate": _KB_PROMPT_TEMPLATE}
                },
            },
        },
    )
    return resp.get("output", {}).get("text", "")


def _call_agent(query: str, session_id: str) -> str:
    """InvokeAgent: KB가 연결된 Bedrock Agent 호출(스트리밍 청크 합침)."""
    resp = _bedrock_agent_runtime.invoke_agent(
        agentId=AGENT_ID,
        agentAliasId=AGENT_ALIAS_ID,
        sessionId=session_id,
        inputText=query,
    )
    chunks = []
    for ev in resp.get("completion", []):
        if "chunk" in ev and "bytes" in ev["chunk"]:
            chunks.append(ev["chunk"]["bytes"].decode("utf-8"))
    return "".join(chunks)


# ---- 모델 출력 파싱 ---------------------------------------------------------
def _parse_guidance(raw_text: str, decision: dict) -> dict:
    """모델이 반환한 텍스트에서 JSON 블록을 안전하게 추출한다."""
    parsed = None
    if raw_text:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                logger.warning("guidance JSON parse failed; using fallback")

    if not isinstance(parsed, dict):
        parsed = _fallback_guidance(decision)

    # 위험도는 확정값으로 강제(모델이 바꿔도 무시)
    parsed["riskLevel"] = decision["riskLevel"]
    parsed.setdefault("headline", f"{decision['lineId']} {decision['riskLevel']}")
    parsed.setdefault("situation", "")
    parsed.setdefault("immediateActions", [])
    parsed.setdefault("followupActions", [])
    parsed.setdefault("citations", [])
    parsed.setdefault("confidence", "LOW")
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


# ---- SNS 통보 ---------------------------------------------------------------
def _publish_sns(decision: dict, guidance: dict) -> bool:
    if not SNS_TOPIC_ARN:
        logger.info("SNS_TOPIC_ARN 미설정 — 발송 생략")
        return False
    if decision["riskLevel"] not in NOTIFY_ON:
        logger.info("riskLevel=%s 은 NOTIFY_ON에 없음 — 발송 생략", decision["riskLevel"])
        return False

    immediate = guidance.get("immediateActions", [])
    body_lines = [
        f"[{decision['riskLevel']}] {guidance.get('headline', '')}",
        f"계류삭 {decision['lineId']} / 선박 {decision['vesselId']} / 사이트 {decision['siteId']}",
        "",
        guidance.get("situation", ""),
    ]
    if immediate:
        body_lines.append("")
        body_lines.append("[즉시 조치]")
        body_lines.extend(f"- {a}" for a in immediate)

    _sns.publish(
        TopicArn=SNS_TOPIC_ARN,
        Subject=f"[AE-Sentinel] {decision['riskLevel']} {decision['lineId']}"[:100],
        Message="\n".join(body_lines),
        MessageAttributes={
            "riskLevel": {"DataType": "String", "StringValue": decision["riskLevel"]},
            "lineId": {"DataType": "String", "StringValue": decision["lineId"]},
        },
    )
    logger.info("SNS 발송 완료: %s", decision["lineId"])
    return True


# ---- 엔트리포인트 -----------------------------------------------------------
def handler(event, context):
    logger.info("event=%s", json.dumps(event, ensure_ascii=False)[:2000])
    decision = _extract_decision(event)
    query = _build_query(decision)

    raw_text = ""
    try:
        if GUIDANCE_MODE == "AGENT":
            raw_text = _call_agent(query, session_id=decision["traceId"])
        else:
            raw_text = _call_kb(query)
    except Exception as exc:  # noqa: BLE001 - PoC 안전망: 실패해도 안내 보장
        logger.exception("Bedrock 호출 실패, fallback 사용: %s", exc)

    logger.info("bedrock raw_text=%s", (raw_text or "")[:1500])
    guidance = _parse_guidance(raw_text, decision)
    alerted = _publish_sns(decision, guidance)

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
        "source": GUIDANCE_MODE,
        "alertDispatched": alerted,
    }
    logger.info("result=%s", json.dumps(result, ensure_ascii=False)[:2000])
    return result
