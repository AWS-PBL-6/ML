"""Port-domain synthesis: underwater lab AE → mooring-eye piezo dataset.

Transforms the Bashir et al. 2017 기반 합성 데이터셋(수중 하이드로폰, dB re 1 uPa)
을 우리 배포 시나리오 — **계류삭의 선박 측 연결 고리(아이·스플라이스 인근)에
부착한 접촉식 피에조 센서** —
도메인으로 변환하고, 실항만 배경 소음·기상 교란·검출 임계 손실을 주입한다.

사용법:
    PYTHONPATH=<repo>/.vendor_ml python3 port_domain_synthesis.py \
        --src ~/Downloads/synthetic_rope_damage_classification_30k.csv \
        --out ~/Downloads/port_mooring_eye_ae_dataset.csv

근거 및 가정 (각 항목은 코드의 PARAMS 에 대응):
  [A1] 접촉식 AE 계측 스케일: dB_AE(re 1 uV), 표준 검출 임계 ~40 dB_AE,
       옥외/소음 환경에서 45~60 dB_AE 로 상향 운용 (ASNT AE field testing;
       MDPI Sensors 20(24):7272). → 임계 = 노이즈 플로어 + 6 dB 부동 임계.
  [A2] 로프 파단 이벤트의 접촉식 AE 진폭: 와이어로프 인장 파단 신호 80~100 dB_AE
       (Casey & Laura 1997; elevator wire rope AE 연구). 섬유(폴리에스터)는
       저에너지 손상기구가 많아 하한이 더 낮음 → 4개 신호군을
       High 85±6 / Medium 62±5 / Low 48±3 / LHF 46±4 dB_AE 로 사상(순서 보존,
       군내 상대편차는 원본 z-score 유지).
  [A3] 합성섬유 로프는 점탄성 재질로 감쇠가 큼(guided-wave 문헌 공통 서술;
       정량치 부재) → 주파수 의존 감쇠 α = 1.0~2.8 dB/m 가정 밴드 +
       계류삭-센서 클램프 커플링 손실 1.5±0.5 dB. **가정임을 명시**.
  [A4] 항만 배경 소음은 갠트리 크레인·정박선 보조엔진·트럭이 지배, 에너지는
       주로 2.5 kHz 이하 저주파 (Port of Long Beach noise map; MDPI
       Sustainability 12(20):8742, 12(5):1740). 접촉식 AE 는 20 kHz 하이패스로
       대부분 차단되나 구조 전파 성분이 플로어를 올림 → 크레인 +10 dB,
       강우 +0.5 dB/(mm/h) (≤+12), 강풍 +0.8 dB/(m/s>6), 정박선 보조기기 +4 dB.
  [A5] 옥외 AE 의 대표적 교란원은 강우 타격·바람·기계 충격 (ASNT; patsnap AE
       interference review) → 위장(spurious) 이벤트로 주입: 강우(다타격·저진폭),
       크레인 충격(중대역·장지속), 바람 마찰(저진폭 단발).
  [A6] 실계류삭 규격: 항만용 폴리에스터 64~88 mm, MBL ≈ 0.25·d² kN 근사
       (Bridon/Samson 카탈로그 근사) — AE 서명은 하중비(%MBL) 기준 불변 가정
       (Bashir: 시편 간 서명 일관 → 외삽 가정임을 명시).
  [A7] 주파수 대역: 수중 계측(≤48 kHz, 96 kHz 샘플링 한계)을 접촉식 광대역
       센서(20~200 kHz, 20 kHz HPF) 대역으로 재사상. LHF 군(0.05~10 kHz)은
       HPF 에 대부분 걸러져 잔존 성분만 약하게 검출.

출력 컬럼:
  관측 가능(학습 피처 후보): Session_ID, Time_Minutes, Rope_Diameter_mm,
    Rope_MBL_kN, AE_Signal_Type, Amplitude_dB_AE, Freq_Low_kHz, Freq_High_kHz,
    Duration_ms, Hit_Count, SNR_dB, Ambient_Noise_dB, Rain_mmh, Wind_mps,
    Crane_Active, Temperature_C, Humidity_pct
  생성 진실값(학습 피처 금지·분석용): Truth_Source, Truth_Distance_m,
    Truth_Load_MBL_Ratio
  라벨: DamageType (안전/주의/위험)
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

PARAMS = {
    # [A2] 신호군 → 소스레벨(dB_AE) 사상 (mu, sigma)
    # 저진폭 마찰 AE 의 발생지는 아이·스플라이스/채핑 존 — 연결 고리 센서와 근접하므로
    # 근거리 검출을 전제로 54 dB_AE 로 둔다 (군간 순서는 그대로 보존).
    "family_dbae": {
        "Low amplitude": (54.0, 4.0),
        "Low to high frequency": (46.0, 4.0),
        "Medium amplitude": (62.0, 5.0),
        "High amplitude": (85.0, 6.0),
    },
    # [A3] 주파수 의존 감쇠 dB/m
    "alpha_db_per_m": {
        "Low amplitude": 1.5,
        "Low to high frequency": 1.0,
        "Medium amplitude": 2.2,
        "High amplitude": 2.8,
    },
    "sensor_coupling_loss": (1.5, 0.5),  # [A3] 로프→클램프형 피에조 센서 전달 손실
    # [A7] 접촉식 센서 대역(kHz): family → (low, high_mu, high_sigma)
    "family_band": {
        "Low amplitude": (20.0, 40.0, 5.0),
        "Low to high frequency": (20.0, 30.0, 4.0),
        "Medium amplitude": (20.0, 120.0, 15.0),
        "High amplitude": (20.0, 180.0, 15.0),
    },
    "duration_ms": {  # (min, max) 균등
        "Low amplitude": (0.3, 1.2),
        "Low to high frequency": (1.0, 4.0),
        "Medium amplitude": (0.5, 2.5),
        "High amplitude": (1.0, 6.0),
    },
    # [A1]/[A4] 노이즈 플로어 모델
    "floor_base": 34.0,
    "floor_crane": 10.0,
    "floor_rain_per_mmh": 0.5,
    "floor_rain_cap": 12.0,
    "floor_wind_per_mps": 0.8,  # v > 6 m/s 초과분
    "floor_ship_aux": 4.0,
    "threshold_over_floor": 6.0,
    # [A5] 교란 이벤트 발생률
    "rain_events_per_min_per_mmh": 0.25,
    "crane_events_per_min": 0.4,
    "wind_events_per_min_per_mps": 0.1,  # v > 8 m/s 초과분
    # 세션 기상/작업 시나리오
    "p_rain_session": 0.35,
    "rain_cover": 0.15,
    "crane_cover": 0.4,
    "seed": 20260709,
}


def _weak_points(rng) -> np.ndarray:
    """세션당 결함 취약점 1~2개의 연결 고리 센서-거리(m).

    마찰·채핑 손상은 아이·스플라이스/페어리드 근방에 집중되므로 지수분포로
    근거리 편향(평균 ~1.4 m, 최대 6 m)을 준다.
    """
    n = rng.integers(1, 3)
    return np.clip(rng.exponential(1.2, size=n) + 0.2, 0.2, 6.0)


def _windows(total: float, cover: float, rng, n_min=1, n_max=3):
    """총 시간(total 분) 중 cover 비율을 덮는 on-구간 리스트 [(t0,t1)...]"""
    n = rng.integers(n_min, n_max + 1)
    span = total * cover / n
    starts = np.sort(rng.uniform(0, total - span, size=n))
    return [(s, s + span) for s in starts]


def _in_windows(t, wins) -> bool:
    return any(a <= t <= b for a, b in wins)


def synthesize(src: pd.DataFrame, params=PARAMS) -> pd.DataFrame:
    rng = np.random.default_rng(params["seed"])
    src = src.drop_duplicates().sort_values(["Trial_ID", "Time_Minutes"]).reset_index(drop=True)

    # 군내 z-score (원본 진폭의 상대 편차 보존 → [A2] 재투영에 사용)
    src["_z"] = src.groupby("AE_Signal_Type")["Amplitude_dB"].transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-9)
    )

    rows = []
    for sid, g in src.groupby("Trial_ID", sort=False):
        g = g.reset_index(drop=True)
        total_min = float(g["Time_Minutes"].max())
        # [A6] 실계류삭 규격
        dia = float(rng.choice([64, 72, 80, 88]))
        mbl = round(0.25 * dia**2 / 10) * 10
        # 세션 기상·작업 시나리오
        rain_wins = _windows(total_min, params["rain_cover"], rng) if rng.random() < params["p_rain_session"] else []
        rain_mmh = float(rng.uniform(2, 20)) if rain_wins else 0.0
        crane_wins = _windows(total_min, params["crane_cover"], rng, 2, 4)
        wind_base = float(rng.uniform(2, 12))
        ship_aux = bool(rng.random() < 0.7)  # 접안 중 보조기기 가동
        temp0 = float(rng.uniform(8, 28))
        hum0 = float(rng.uniform(45, 85))
        weak = _weak_points(rng)

        def env_at(t):
            rain = rain_mmh if _in_windows(t, rain_wins) else 0.0
            wind = max(0.5, wind_base + 2.5 * np.sin(t / 180) + rng.normal(0, 0.8))
            crane = _in_windows(t, crane_wins)
            floor = (
                params["floor_base"]
                + (params["floor_crane"] if crane else 0.0)
                + min(params["floor_rain_cap"], params["floor_rain_per_mmh"] * rain)
                + params["floor_wind_per_mps"] * max(0.0, wind - 6)
                + (params["floor_ship_aux"] if ship_aux else 0.0)
                + rng.normal(0, 1.5)
            )
            temp = temp0 + 3 * np.sin(t / 240) + rng.normal(0, 0.4)
            hum = min(98, hum0 + (10 if rain else 0) + rng.normal(0, 2))
            return rain, wind, crane, floor, temp, hum

        def emit(t, fam, amp_src, hits, dur, dist, source, ratio, label):
            rain, wind, crane, floor, temp, hum = env_at(t)
            thr = floor + params["threshold_over_floor"]
            alpha = params["alpha_db_per_m"][fam]
            coupling = rng.normal(*params["sensor_coupling_loss"])
            received = amp_src - alpha * dist - coupling + rng.normal(0, 1.0)
            if source != "ROPE":  # 강우/마찰/충격이 센서 클램프 주변에 직접 유입
                received = amp_src
            if received < thr:
                return  # [A1] 검출 실패 (미탐)
            lo, hi_mu, hi_sd = params["family_band"][fam]
            hi = max(lo + 5, rng.normal(hi_mu, hi_sd))
            rows.append({
                "Session_ID": sid.replace("Trial", "Session"),
                "Time_Minutes": round(float(t), 1),
                "Rope_Diameter_mm": dia,
                "Rope_MBL_kN": mbl,
                "AE_Signal_Type": fam,
                "Amplitude_dB_AE": round(float(received), 1),
                "Freq_Low_kHz": lo,
                "Freq_High_kHz": round(float(hi), 1),
                "Duration_ms": round(float(dur), 2),
                "Hit_Count": int(hits),
                "SNR_dB": round(float(received - floor), 1),
                "Ambient_Noise_dB": round(float(floor), 1),
                "Rain_mmh": round(float(rain), 1),
                "Wind_mps": round(float(wind), 1),
                "Crane_Active": int(crane),
                "Temperature_C": round(float(temp), 1),
                "Humidity_pct": round(float(hum), 1),
                "Truth_Source": source,
                "Truth_Distance_m": round(float(dist), 2),
                "Truth_Load_MBL_Ratio": round(float(ratio), 3),
                "DamageType": label,
            })

        # ── 로프 AE (원본 행 변환) ─────────────────────────────────────────
        for _, r in g.iterrows():
            fam = r["AE_Signal_Type"]
            mu, sd = params["family_dbae"][fam]
            amp_src = mu + float(r["_z"]) * sd  # 군내 상대 편차 보존
            # 손상 위치: 중/고진폭은 취약점, 저진폭은 60% 취약점 근방 / 40% 임의
            if fam in ("Medium amplitude", "High amplitude") or rng.random() < 0.6:
                dist = float(rng.choice(weak)) + rng.normal(0, 0.3)
            else:
                dist = rng.uniform(0.3, 8.0)
            dist = float(np.clip(dist, 0.2, 9.0))
            dmin, dmax = params["duration_ms"][fam]
            emit(r["Time_Minutes"], fam, amp_src, r["Hit_Count"],
                 rng.uniform(dmin, dmax), dist, "ROPE",
                 r["Load_MBL_Ratio"], r["DamageType"])

        # ── 교란(spurious) 이벤트 주입 [A5] ────────────────────────────────
        # 시각 t 의 라벨 = 그 시각 로프 상태 (시간축 보간)
        tt = g["Time_Minutes"].values
        ll = g["DamageType"].values
        rr = g["Load_MBL_Ratio"].values
        def state_at(t):
            i = int(np.clip(np.searchsorted(tt, t), 0, len(tt) - 1))
            return ll[i], rr[i]

        for a, b in rain_wins:
            n = rng.poisson(params["rain_events_per_min_per_mmh"] * rain_mmh * (b - a))
            for t in rng.uniform(a, b, size=n):
                lab, ratio = state_at(t)
                _, _, _, floor, _, _ = env_at(t)
                emit(t, "Low amplitude", floor + rng.normal(9, 2),
                     rng.integers(3, 13), rng.uniform(0.05, 0.4), 0.0, "RAIN", ratio, lab)
        for a, b in crane_wins:
            n = rng.poisson(params["crane_events_per_min"] * (b - a))
            for t in rng.uniform(a, b, size=n):
                lab, ratio = state_at(t)
                _, _, _, floor, _, _ = env_at(t)
                emit(t, "Medium amplitude", floor + rng.normal(12, 3),
                     rng.integers(2, 7), rng.uniform(5, 40), 0.0, "CRANE", ratio, lab)
        if wind_base > 8:
            n = rng.poisson(params["wind_events_per_min_per_mps"] * (wind_base - 8) * total_min)
            for t in rng.uniform(0, total_min, size=n):
                lab, ratio = state_at(t)
                _, _, _, floor, _, _ = env_at(t)
                emit(t, "Low to high frequency", floor + rng.normal(7, 1.5),
                     rng.integers(1, 3), rng.uniform(1, 4), 0.0, "WIND", ratio, lab)

    out = pd.DataFrame(rows).sort_values(["Session_ID", "Time_Minutes"]).reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    src = pd.read_csv(args.src)
    out = synthesize(src)
    out.to_csv(args.out, index=False)
    kept = (out["Truth_Source"] == "ROPE").sum()
    print(f"입력 로프 이벤트 {len(src)} → 검출 {kept} (미탐 {1 - kept/len(src):.1%})")
    print(f"교란 이벤트 {len(out) - kept} | 총 {len(out)} rows → {args.out}")
    print(out["DamageType"].value_counts().to_string())
    print(out["Truth_Source"].value_counts().to_string())


if __name__ == "__main__":
    main()
