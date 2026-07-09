"""Shared window-feature contract for the port-domain AE breakage model.

The model does NOT classify a single AE event — it classifies the **recent
event stream** on one mooring line. This module defines that windowing so the
training pipeline and the serving side (backend inference_service assembling
`repo.list_events(lineId, limit=W)`) compute *identical* features.

Contract (frozen — bump WINDOW_SPEC_VERSION on any change):
  - Order events on a line by time ascending.
  - For the newest event, aggregate the last ``WINDOW_SIZE`` events
    (inclusive), requiring at least ``MIN_EVENTS`` to emit a prediction.
  - Feature order is ``WINDOW_FEATURES`` exactly.
"""
from __future__ import annotations

from typing import List, Sequence

WINDOW_SPEC_VERSION = "1.0.0"
WINDOW_SIZE = 12
MIN_EVENTS = 3

# Ordered feature names — the serving side MUST emit these in this order.
WINDOW_FEATURES: List[str] = [
    "amp_mean", "amp_max",          # Amplitude_dB_AE
    "snr_mean", "snr_max",          # SNR_dB
    "hit_sum", "hit_max",           # Hit_Count
    "dur_max",                      # Duration_ms
    "n_low", "n_med", "n_high", "n_lhf",  # AE_Signal_Type composition
    "fhigh_max",                    # Freq_High_kHz
    "rate",                         # events per minute over the window
    "noise", "rain", "wind", "crane",     # current-event environment covariates
]

# AE_Signal_Type canonical values (composition counts).
SIGNAL_LOW = "Low amplitude"
SIGNAL_LHF = "Low to high frequency"
SIGNAL_MED = "Medium amplitude"
SIGNAL_HIGH = "High amplitude"


class EventView:
    """Minimal per-event fields the window aggregation needs.

    Mirrors both the training CSV columns and the fields the backend pulls
    from an EventRecord's ``features``/``context`` when assembling a window.
    """

    __slots__ = (
        "time_min", "amp", "snr", "hit", "dur", "fhigh",
        "signal_type", "noise", "rain", "wind", "crane",
    )

    def __init__(self, time_min, amp, snr, hit, dur, fhigh, signal_type,
                 noise, rain, wind, crane):
        self.time_min = float(time_min)
        self.amp = float(amp)
        self.snr = float(snr)
        self.hit = float(hit)
        self.dur = float(dur)
        self.fhigh = float(fhigh)
        self.signal_type = str(signal_type)
        self.noise = float(noise)
        self.rain = float(rain)
        self.wind = float(wind)
        self.crane = float(crane)


def window_vector(events: Sequence[EventView]) -> List[float] | None:
    """Compute the ordered feature vector for the newest event in ``events``.

    ``events`` must be time-ascending; the last element is the event being
    scored. Returns ``None`` when fewer than ``MIN_EVENTS`` are available.
    """
    if len(events) < MIN_EVENTS:
        return None
    win = list(events[-WINDOW_SIZE:])
    cur = win[-1]

    amps = [e.amp for e in win]
    snrs = [e.snr for e in win]
    hits = [e.hit for e in win]
    durs = [e.dur for e in win]
    fhis = [e.fhigh for e in win]

    n_low = sum(1 for e in win if e.signal_type == SIGNAL_LOW)
    n_med = sum(1 for e in win if e.signal_type == SIGNAL_MED)
    n_high = sum(1 for e in win if e.signal_type == SIGNAL_HIGH)
    n_lhf = sum(1 for e in win if e.signal_type == SIGNAL_LHF)

    span = cur.time_min - win[0].time_min
    rate = len(win) / span if span > 1e-6 else float(len(win))

    values = {
        "amp_mean": sum(amps) / len(amps),
        "amp_max": max(amps),
        "snr_mean": sum(snrs) / len(snrs),
        "snr_max": max(snrs),
        "hit_sum": sum(hits),
        "hit_max": max(hits),
        "dur_max": max(durs),
        "n_low": float(n_low),
        "n_med": float(n_med),
        "n_high": float(n_high),
        "n_lhf": float(n_lhf),
        "fhigh_max": max(fhis),
        "rate": rate,
        "noise": cur.noise,
        "rain": cur.rain,
        "wind": cur.wind,
        "crane": cur.crane,
    }
    return [float(values[name]) for name in WINDOW_FEATURES]
