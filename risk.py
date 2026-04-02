import numpy as np


THRESHOLDS = {
    "critical": {"rt": 2.0,  "growth": 0.30, "risk": 0.75, "momentum": 1.5},
    "high":     {"rt": 1.2,  "growth": 0.10, "risk": 0.50, "momentum": 0.5},
    "moderate": {"rt": 1.0,  "growth": 0.00, "risk": 0.25, "momentum": 0.0},
}

RISK_TIERS = {
    "Critical": {"color": "#ef4444", "bg": "alert-critical",  "icon": "🔴"},
    "High":     {"color": "#f97316", "bg": "alert-high",      "icon": "🟠"},
    "Moderate": {"color": "#eab308", "bg": "alert-moderate",  "icon": "🟡"},
    "Low":      {"color": "#22c55e", "bg": "alert-controlled","icon": "🟢"},
}


def classify_risk(risk_score: float) -> str:
    if risk_score >= THRESHOLDS["critical"]["risk"]: return "Critical"
    if risk_score >= THRESHOLDS["high"]["risk"]:     return "High"
    if risk_score >= THRESHOLDS["moderate"]["risk"]: return "Moderate"
    return "Low"


def get_alert(
    rt: float,
    growth: float,
    risk_score: float,
    momentum: float = 0.0,
    rt_trend: float = 0.0,
    healthcare_pressure: float = 0.0,
) -> dict:
    """
    Multi-signal alert combining Rt, growth, momentum, Rt trend, and healthcare pressure.
    Returns a dict with level, CSS class, narrative message, and tier label.
    """
    tier = classify_risk(risk_score)

    if (rt >= THRESHOLDS["critical"]["rt"]
            or (rt > THRESHOLDS["high"]["rt"] and growth > THRESHOLDS["critical"]["growth"])
            or momentum > THRESHOLDS["critical"]["momentum"]):
        level   = "🔴 Critical"
        message = (f"Exponential spread — Rt={rt:.2f}, momentum={momentum:.2f}, "
                   f"weekly growth={growth*100:.1f}%")

    elif rt > THRESHOLDS["high"]["rt"] and growth > THRESHOLDS["high"]["growth"]:
        level   = "🟠 High"
        message = f"Rapid transmission — Rt={rt:.2f} with accelerating case counts"

    elif rt > THRESHOLDS["moderate"]["rt"]:
        level = "🟡 Moderate"
        if rt_trend > 0.02:
            trend_note = "and accelerating"
        elif rt_trend < -0.02:
            trend_note = "but stabilising"
        else:
            trend_note = ""
        message = f"Growing transmission — Rt={rt:.2f} {trend_note}".strip()

    else:
        level   = "🟢 Controlled"
        message = f"Outbreak appears controlled — Rt={rt:.2f}"

    if healthcare_pressure > 0.6 and tier in ("High", "Critical"):
        message += f" | Healthcare pressure at {healthcare_pressure*100:.0f}% of historical peak"

    return {
        "level":   level,
        "color":   RISK_TIERS[tier]["bg"],
        "message": message,
        "tier":    tier,
    }


def get_risk_drivers(
    rt: float,
    growth: float,
    cfr: float,
    doubling_time: float,
    momentum: float = 0.0,
    rt_trend: float = 0.0,
    incidence_per_100k: float = 0.0,
    healthcare_pressure_norm: float = 0.0,
    death_acceleration: float = 0.0,
    vacc_fully: float = 0.0,
    test_positivity_rate: float = 0.0,
) -> list[str]:
    """
    Comprehensive risk driver analysis with vaccination and testing context.
    Returns ordered list of plain-English explanations for public health decision-makers.
    Ordering: transmission → severity → burden → contextual mitigators.
    """
    drivers = []

    # 1. Rt level and direction
    if rt > 2.0:
        drivers.append(f"🔴 Critical Rt ({rt:.2f}) — each case generating >2 secondary infections, exponential phase")
    elif rt > 1.5:
        drivers.append(f"🟠 High Rt ({rt:.2f}) — each case generating >1.5 secondary infections, rapid expansion")
    elif rt > 1.0:
        drivers.append(f"🟡 Rt above threshold ({rt:.2f}) — outbreak in expansion phase")

    if rt_trend > 0.05:
        drivers.append(f"📈 Rt rising (+{rt_trend:.3f}/day) — transmission accelerating, not stabilising")
    elif rt_trend < -0.05:
        drivers.append(f"📉 Rt falling ({rt_trend:.3f}/day) — transmission decelerating, positive signal")

    # 2. Transmission momentum
    if momentum > 1.5:
        drivers.append(f"⚡ Momentum critical ({momentum:.2f}) — high Rt AND fast growth rate simultaneously")
    elif momentum > 0.5:
        drivers.append(f"⚠️ Momentum elevated ({momentum:.2f}) — speed and intensity both above normal")

    # 3. Growth and doubling time
    if growth > 0.30:
        drivers.append(f"🔴 Extreme weekly growth ({growth*100:.1f}%) — case count near-doubling weekly")
    elif growth > 0.15:
        drivers.append(f"🟠 High weekly growth ({growth*100:.1f}%) — significant case acceleration")
    elif growth > 0.05:
        drivers.append(f"🟡 Moderate weekly growth ({growth*100:.1f}%)")

    if 0 < doubling_time < 5:
        drivers.append(f"🔴 Cases doubling every {doubling_time:.1f} days — explosive spread")
    elif 5 <= doubling_time < 10:
        drivers.append(f"🟠 Cases doubling every {doubling_time:.1f} days — fast spread")
    elif 10 <= doubling_time < 21:
        drivers.append(f"🟡 Cases doubling every {doubling_time:.1f} days — moderate spread")

    # 4. CFR and severity
    if cfr > 0.05:
        drivers.append(f"🔴 High CFR ({cfr*100:.1f}%) — significant mortality, possible healthcare saturation")
    elif cfr > 0.02:
        drivers.append(f"🟡 Elevated CFR ({cfr*100:.1f}%) — above typical IFR, monitor for system stress")

    # 5. Healthcare pressure
    if healthcare_pressure_norm > 0.8:
        drivers.append(f"🔴 Healthcare pressure at {healthcare_pressure_norm*100:.0f}% of historical peak — near capacity")
    elif healthcare_pressure_norm > 0.5:
        drivers.append(f"🟠 Healthcare pressure elevated ({healthcare_pressure_norm*100:.0f}% of peak)")

    # 6. Incidence per 100k (WHO-normalised)
    if incidence_per_100k > 500:
        drivers.append(f"🔴 Incidence {incidence_per_100k:.0f}/100k — far above WHO threshold (>50)")
    elif incidence_per_100k > 50:
        drivers.append(f"🟠 Incidence {incidence_per_100k:.0f}/100k — above WHO high-transmission threshold")
    elif incidence_per_100k > 10:
        drivers.append(f"🟡 Incidence {incidence_per_100k:.1f}/100k — moderate community transmission")

    # 7. Death acceleration (early warning signal)
    if death_acceleration > 0.20:
        drivers.append(
            f"⚠️ Deaths accelerating (+{death_acceleration*100:.1f}%/week) — "
            f"severity surge signal, may indicate a peak is still 1–2 weeks away"
        )

    # 8. Testing context — underdetection warning
    # WHO: positivity > 5% means case counts are an underestimate of true spread
    if test_positivity_rate > 0:
        if test_positivity_rate > 0.20:
            drivers.append(
                f"🔴 Test positivity {test_positivity_rate*100:.1f}% — severe underdetection likely, "
                f"true case burden substantially higher than reported"
            )
        elif test_positivity_rate > 0.05:
            drivers.append(
                f"🟡 Test positivity {test_positivity_rate*100:.1f}% — above WHO 5% threshold, "
                f"case counts may underestimate true spread"
            )
        else:
            drivers.append(
                f"✅ Test positivity {test_positivity_rate*100:.1f}% — adequate testing coverage, "
                f"case counts are a reliable signal"
            )

    # 9. Vaccination context — mitigating or absent protection
    if vacc_fully > 0:
        if vacc_fully >= 70:
            drivers.append(
                f"💉 {vacc_fully:.0f}% fully vaccinated — high population protection; "
                f"rising Rt may reflect waning immunity or immune-evasive variant"
            )
        elif vacc_fully >= 40:
            drivers.append(
                f"💉 {vacc_fully:.0f}% fully vaccinated — partial coverage; "
                f"unvaccinated population remains a transmission reservoir"
            )
        else:
            drivers.append(
                f"💉 {vacc_fully:.0f}% fully vaccinated — low coverage; "
                f"population largely immunologically naive, amplifying outbreak risk"
            )

    if not drivers:
        drivers.append("✅ No significant risk drivers — outbreak appears controlled across all indicators")

    return drivers
