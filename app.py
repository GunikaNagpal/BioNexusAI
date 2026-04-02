import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

from forecast import forecast_cases
from risk import classify_risk, get_alert, get_risk_drivers, RISK_TIERS
from spatial import compute_mobility_rt_correlation, detect_hotspots

ROOT          = Path(__file__).resolve().parent.parent
DATA_PATH     = ROOT / "data" / "processed" / "risk.csv"
MOBILITY_PATH = ROOT / "data" / "processed" / "mobility.csv"
HOTSPOT_PATH  = ROOT / "data" / "processed" / "hotspots.csv"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EpiIQ Sentinel",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: #0a0e1a;
    color: #e2e8f0;
}
.stApp { background-color: #0a0e1a; }

[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0f1629 0%, #131d35 100%);
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}
[data-testid="metric-container"]:hover {
    border-color: #2563eb;
    box-shadow: 0 4px 32px rgba(37,99,235,0.15);
    transition: all 0.2s ease;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.9rem !important;
    font-weight: 600;
    color: #60a5fa;
}
[data-testid="stMetricLabel"] {
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748b;
}
[data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
}
.section-header {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #2563eb;
    margin: 2rem 0 1rem;
    border-bottom: 1px solid #1e2d4a;
    padding-bottom: 0.5rem;
}
.alert-critical  { background: linear-gradient(135deg,#450a0a,#1a0000); border-left:4px solid #ef4444; border-radius:8px; padding:16px 20px; margin:1rem 0; }
.alert-high      { background: linear-gradient(135deg,#431407,#1a0800); border-left:4px solid #f97316; border-radius:8px; padding:16px 20px; margin:1rem 0; }
.alert-moderate  { background: linear-gradient(135deg,#422006,#1a1000); border-left:4px solid #eab308; border-radius:8px; padding:16px 20px; margin:1rem 0; }
.alert-controlled{ background: linear-gradient(135deg,#052e16,#001a0a); border-left:4px solid #22c55e; border-radius:8px; padding:16px 20px; margin:1rem 0; }
.driver-item { background:#0f1629; border:1px solid #1e2d4a; border-radius:8px; padding:10px 14px; margin:5px 0; font-size:0.9rem; color:#94a3b8; }
.driver-item:hover { border-color:#2563eb; transition:border-color 0.2s; }
.main-title    { font-size:2.4rem; font-weight:700; color:#f1f5f9; letter-spacing:-0.02em; }
.main-subtitle { font-size:0.95rem; color:#475569; margin-top:0.2rem; letter-spacing:0.04em; }
.metric-sub { font-size:0.72rem; color:#475569; text-align:center; margin-top:2px; font-family:'Space Grotesk'; }
[data-testid="stSidebar"] { background-color:#0d1220; border-right:1px solid #1e2d4a; }
[data-testid="stDataFrame"] { background:#0f1629; border-radius:12px; }
.stSelectbox label { color:#64748b; font-size:0.8rem; letter-spacing:0.08em; text-transform:uppercase; }
</style>
""", unsafe_allow_html=True)

# ── Plotly base layout ────────────────────────────────────────────────────────
PL = dict(
    paper_bgcolor="#0a0e1a", plot_bgcolor="#0f1629",
    font=dict(family="Space Grotesk", color="#94a3b8"),
    xaxis=dict(gridcolor="#1e2d4a", zerolinecolor="#1e2d4a", linecolor="#1e2d4a"),
    yaxis=dict(gridcolor="#1e2d4a", zerolinecolor="#1e2d4a", linecolor="#1e2d4a"),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(bgcolor="#0f1629", bordercolor="#1e2d4a", borderwidth=1),
)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["date"])

@st.cache_data(ttl=3600)
def load_mobility_data():
    return pd.read_csv(MOBILITY_PATH, parse_dates=["date"]) if MOBILITY_PATH.exists() else None

@st.cache_data(ttl=3600)
def load_hotspots():
    return pd.read_csv(HOTSPOT_PATH) if HOTSPOT_PATH.exists() else None


# ── Helpers ───────────────────────────────────────────────────────────────────
def line(df, x, y, name, color, dash="solid", width=2):
    return go.Scatter(x=df[x], y=df[y], mode="lines", name=name,
                      line=dict(color=color, width=width, dash=dash))

def safe_val(series, default=0.0):
    v = series.iloc[-1] if len(series) > 0 else default
    return float(v) if not (np.isnan(v) or np.isinf(v)) else default


# ── Header ────────────────────────────────────────────────────────────────────
_, col_title = st.columns([1, 8])
with col_title:
    st.markdown('<div class="main-title">🧬 EpiIQ Sentinel</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">AI-Powered Epidemic Intelligence System</div>', unsafe_allow_html=True)
st.markdown("---")

# ── Guard ─────────────────────────────────────────────────────────────────────
if not DATA_PATH.exists():
    st.error("⚠️ No data found. Run `pipeline.py` first.")
    st.stop()

df         = load_data()
mob_df     = load_mobility_data()
hotspot_df = load_hotspots()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌍 Country")
    countries = sorted(df["country"].unique())
    default   = countries.index("India") if "India" in countries else 0
    country   = st.selectbox("Select Country", countries, index=default)

    st.markdown("---")
    st.markdown("### ℹ️ Methodology")
    st.markdown("""
**Rt** · Ratio method, SI=7d, clipped [0,5]

**Risk Score (0–1)**
- 30% Rt component
- 20% Weekly growth
- 20% Lagged CFR (14d)
- 10% Relative incidence
- 10% Transmission momentum
- 10% Healthcare pressure

**Momentum** · Rt × growth rate

**Healthcare Pressure** · cases × CFR, normalised to own peak

**Wave Breaks** · CUSUM structural break detection

**Rt Trend** · 7-day rolling linear slope of Rt

**Doubling Time** · ln(2)/ln(1+g)

**Incidence/100k** · WHO standard normalisation

**Forecast** · Damped exponential, 80% CI

**Population** · OWID (auto-fetched, no hardcoding)

**Vaccination** · OWID people_fully_vaccinated_per_hundred

**Testing** · OWID positive_rate (WHO threshold: 5%)
    """)
    st.markdown("---")
    st.markdown(f"**Data:** JHU CSSE  \n**Countries:** {df['country'].nunique()}")

# ── Filter to country ─────────────────────────────────────────────────────────
cdf    = df[df["country"] == country].copy().sort_values("date")
latest = cdf.iloc[-1]

rt_val          = round(safe_val(cdf["Rt"]), 2)
growth_val      = round(safe_val(cdf["growth_rate"]), 4)
risk_val        = round(safe_val(cdf["risk_score"]), 3)
cfr_val         = round(safe_val(cdf["CFR"]) * 100, 2)
dt_val          = safe_val(cdf["doubling_time"])
momentum_val    = round(safe_val(cdf["transmission_momentum"]), 3)
rt_trend_val    = round(safe_val(cdf["Rt_trend"]), 4)
pressure_val    = round(safe_val(cdf["healthcare_pressure_norm"]), 3)
incidence_100k  = round(safe_val(cdf.get("incidence_per_100k", pd.Series([0]))), 1)
death_acc_val   = round(safe_val(cdf["death_acceleration"]), 4)
new_cases_val   = int(safe_val(cdf["new_cases"]))

alert = get_alert(
    rt_val, growth_val, risk_val,
    momentum=momentum_val, rt_trend=rt_trend_val,
    healthcare_pressure=pressure_val
)
# Vaccination and testing — use latest available non-zero value
vacc_fully_val       = round(safe_val(cdf.get("vacc_fully",       pd.Series([0]))), 1)
test_positivity_val  = round(safe_val(cdf.get("test_positivity_rate", pd.Series([0]))), 4)

drivers = get_risk_drivers(
    rt_val, growth_val, safe_val(cdf["CFR"]), dt_val,
    momentum=momentum_val, rt_trend=rt_trend_val,
    incidence_per_100k=incidence_100k,
    healthcare_pressure_norm=pressure_val,
    death_acceleration=death_acc_val,
    vacc_fully=vacc_fully_val,
    test_positivity_rate=test_positivity_val,
)

# ── Alert Banner ──────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="{alert['color']}">
  <strong style="font-size:1.1rem">{alert['level']} — {country}</strong><br>
  <span style="color:#94a3b8;font-size:0.9rem">{alert['message']}</span>
</div>
""", unsafe_allow_html=True)

# ── Metric Row 1: Core transmission signals ───────────────────────────────────
st.markdown('<div class="section-header">Epidemiological Signals</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Effective Rt", rt_val,
          delta="↑ Expanding" if rt_val > 1 else "↓ Shrinking", delta_color="inverse")
c2.metric("Weekly Growth", f"{growth_val*100:.1f}%",
          delta="Accelerating" if growth_val > 0.05 else "Stable", delta_color="inverse")
c3.metric("Risk Score (0–1)", risk_val, delta=classify_risk(risk_val))
c4.metric("Lagged CFR", f"{cfr_val}%",
          help="14-day lag accounts for outcome delay")
c5.metric("Doubling Time",
          f"{dt_val:.1f}d" if dt_val > 0 else "N/A",
          delta="Fast" if 0 < dt_val < 7 else ("Moderate" if dt_val < 21 else "Slow"),
          delta_color="inverse")
c6.metric("Incidence /100k",
          f"{incidence_100k:.1f}" if incidence_100k > 0 else "N/A",
          help="WHO standard: >50 = high transmission")

# ── Metric Row 2: New advanced signals ───────────────────────────────────────
st.markdown('<div class="section-header">Advanced Epidemiological Signals</div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Transmission Momentum",
          f"{momentum_val:.3f}",
          delta="Critical" if momentum_val > 1.5 else ("Elevated" if momentum_val > 0.5 else "Normal"),
          delta_color="inverse",
          help="Rt × growth rate. Captures simultaneous speed and intensity of spread.")
m2.metric("Rt Trend (7d slope)",
          f"{rt_trend_val:+.4f}",
          delta="Accelerating ↑" if rt_trend_val > 0.02 else ("Stabilising ↓" if rt_trend_val < -0.02 else "Flat →"),
          delta_color="inverse",
          help="Positive = Rt rising (worsening); negative = Rt falling (improving)")
m3.metric("Healthcare Pressure",
          f"{pressure_val*100:.0f}%",
          delta="of historical peak",
          delta_color="off",
          help="Smoothed cases × lagged CFR, normalised to country's own historical maximum")
m4.metric("Death Acceleration",
          f"{death_acc_val*100:.1f}%/wk",
          delta="Surge signal" if death_acc_val > 0.2 else "Normal",
          delta_color="inverse",
          help="Week-over-week growth in deaths. Leads case-count decline by ~14 days.")

# ── Risk Drivers ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Risk Driver Analysis</div>', unsafe_allow_html=True)
for d in drivers:
    st.markdown(f'<div class="driver-item">{d}</div>', unsafe_allow_html=True)

# ── SECTION: Transmission Dynamics ───────────────────────────────────────────
st.markdown('<div class="section-header">Transmission Dynamics</div>', unsafe_allow_html=True)

r1c1, r1c2 = st.columns(2)

with r1c1:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=cdf["date"], y=cdf["new_cases"],
                         name="Daily Cases", marker_color="rgba(37,99,235,0.22)"))
    fig.add_trace(line(cdf, "date", "cases_smooth", "7-day Average", "#60a5fa", width=2.5))
    if "incidence_per_100k" in cdf.columns:
        fig.add_trace(go.Scatter(x=cdf["date"], y=cdf["incidence_per_100k"],
                                 mode="lines", name="Incidence/100k",
                                 line=dict(color="#fbbf24", width=1.5, dash="dot"),
                                 yaxis="y2"))
    fig.update_layout(**PL,
        title=dict(text="Daily Cases + Incidence per 100k", font=dict(size=14, color="#e2e8f0")),
        barmode="overlay",
        yaxis2=dict(title="per 100k", overlaying="y", side="right",
                    gridcolor="#1e2d4a", zerolinecolor="#1e2d4a", showgrid=False))
    st.plotly_chart(fig, use_container_width=True)

with r1c2:
    fig = go.Figure()
    fig.add_hline(y=1.0, line_dash="dash", line_color="#475569", annotation_text="Rt = 1")
    fig.add_hline(y=1.5, line_dash="dot",  line_color="#374151")
    # Red shading above Rt=1
    fig.add_trace(go.Scatter(
        x=pd.concat([cdf["date"], cdf["date"][::-1]]),
        y=pd.concat([cdf["Rt"].clip(lower=1), pd.Series(np.ones(len(cdf)))[::-1]]),
        fill="toself", fillcolor="rgba(239,68,68,0.1)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=cdf["date"], y=cdf["Rt"], mode="lines",
                             name="Rt", line=dict(color="#f97316", width=2)))
    # Rt trend overlay
    if "Rt_trend" in cdf.columns:
        rt_trend_smooth = cdf["Rt_trend"].rolling(7, min_periods=3).mean()
        fig.add_trace(go.Scatter(x=cdf["date"], y=rt_trend_smooth,
                                 mode="lines", name="Rt Slope (7d)",
                                 line=dict(color="#a78bfa", width=1.5, dash="dot"),
                                 yaxis="y2"))
    fig.update_layout(**PL,
        title=dict(text="Effective Rt + Rt Trend Slope", font=dict(size=14, color="#e2e8f0")),
        yaxis=dict(**PL["yaxis"], range=[0, min(float(cdf["Rt"].max()) + 0.5, 5)]),
        yaxis2=dict(title="Rt slope", overlaying="y", side="right",
                    gridcolor="#1e2d4a", zerolinecolor="#1e2d4a", showgrid=False))
    st.plotly_chart(fig, use_container_width=True)

# ── SECTION: Momentum & Severity ─────────────────────────────────────────────
st.markdown('<div class="section-header">Transmission Momentum & Severity</div>', unsafe_allow_html=True)

r2c1, r2c2 = st.columns(2)

with r2c1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cdf["date"], y=cdf["transmission_momentum"],
                             mode="lines", name="Transmission Momentum",
                             line=dict(color="#f97316", width=2),
                             fill="tozeroy", fillcolor="rgba(249,115,22,0.1)"))
    for thresh, label, color in [(1.5,"Critical","#ef4444"), (0.5,"Elevated","#f97316")]:
        fig.add_hline(y=thresh, line_dash="dot", line_color=color,
                      annotation_text=label, annotation_font_color=color)
    fig.update_layout(**PL,
        title=dict(text="Transmission Momentum (Rt × Growth Rate)", font=dict(size=14, color="#e2e8f0")))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Captures the combined effect of transmission intensity (Rt) and speed (growth rate). A country with Rt=1.5 and 20% weekly growth has momentum 0.30.")

with r2c2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cdf["date"], y=cdf["healthcare_pressure_norm"] * 100,
                             mode="lines", name="Healthcare Pressure",
                             line=dict(color="#f43f5e", width=2),
                             fill="tozeroy", fillcolor="rgba(244,63,94,0.08)"))
    for thresh, label, color in [(80,"Near Capacity","#ef4444"),(50,"Elevated","#f97316")]:
        fig.add_hline(y=thresh, line_dash="dot", line_color=color,
                      annotation_text=f"{label} ({thresh}%)", annotation_font_color=color)
    fig.update_layout(**PL,
        title=dict(text="Healthcare Pressure (% of Historical Peak)", font=dict(size=14, color="#e2e8f0")),
        yaxis=dict(**PL["yaxis"], range=[0, 105]))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Cases × lagged CFR, normalised to country's own peak. Proxy for relative healthcare system burden.")

# ── SECTION: Risk Score + Growth ─────────────────────────────────────────────
st.markdown('<div class="section-header">Risk Score & Growth</div>', unsafe_allow_html=True)

r3c1, r3c2 = st.columns(2)

with r3c1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cdf["date"], y=cdf["risk_score"],
                             mode="lines", name="Risk Score",
                             line=dict(color="#8b5cf6", width=2),
                             fill="tozeroy", fillcolor="rgba(139,92,246,0.1)"))
    # Wave break markers
    if "wave_break" in cdf.columns:
        breaks = cdf[cdf["wave_break"] == 1]
        if not breaks.empty:
            fig.add_trace(go.Scatter(
                x=breaks["date"], y=breaks["risk_score"],
                mode="markers", name="Wave Break / Regime Shift",
                marker=dict(color="#fbbf24", size=10, symbol="diamond",
                            line=dict(color="#f97316", width=2))))
    for thresh, label, color in [(0.75,"Critical","#ef4444"),(0.5,"High","#f97316"),(0.25,"Moderate","#eab308")]:
        fig.add_hline(y=thresh, line_dash="dot", line_color=color,
                      annotation_text=label, annotation_font_color=color)
    fig.update_layout(**PL,
        title=dict(text="Composite Risk Score with Wave Breaks", font=dict(size=14, color="#e2e8f0")),
        yaxis=dict(**PL["yaxis"], range=[0, 1]))
    st.plotly_chart(fig, use_container_width=True)

with r3c2:
    fig = go.Figure()
    fig.add_hline(y=0, line_color="#475569", line_dash="dash")
    fig.add_trace(go.Scatter(x=cdf["date"], y=cdf["growth_rate"] * 100,
                             mode="lines", name="Weekly Growth %",
                             line=dict(color="#06b6d4", width=2)))
    # Positive shading
    pos = cdf["growth_rate"] * 100
    fig.add_trace(go.Scatter(
        x=pd.concat([cdf["date"], cdf["date"][::-1]]),
        y=pd.concat([pos.clip(lower=0), pd.Series(np.zeros(len(cdf)))[::-1]]),
        fill="toself", fillcolor="rgba(239,68,68,0.07)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip"))
    fig.update_layout(**PL,
        title=dict(text="Week-over-Week Growth Rate (%)", font=dict(size=14, color="#e2e8f0")))
    st.plotly_chart(fig, use_container_width=True)

# ── SECTION: CFR & Deaths ─────────────────────────────────────────────────────
st.markdown('<div class="section-header">Severity & Mortality</div>', unsafe_allow_html=True)

r4c1, r4c2 = st.columns(2)

with r4c1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cdf["date"], y=cdf["CFR"] * 100,
                             mode="lines", name="CFR % (14d lag)",
                             line=dict(color="#f43f5e", width=2),
                             fill="tozeroy", fillcolor="rgba(244,63,94,0.08)"))
    fig.add_hline(y=2, line_dash="dot", line_color="#374151",
                  annotation_text="2% reference", annotation_font_color="#64748b")
    fig.update_layout(**PL,
        title=dict(text="Case Fatality Ratio — 14-day lagged (%)", font=dict(size=14, color="#e2e8f0")))
    st.plotly_chart(fig, use_container_width=True)

with r4c2:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=cdf["date"], y=cdf["new_deaths"],
                         name="Daily Deaths", marker_color="rgba(244,63,94,0.25)"))
    fig.add_trace(line(cdf, "date", "deaths_smooth", "7-day Average", "#f43f5e", width=2))
    # Death acceleration overlay
    if "death_acceleration" in cdf.columns:
        fig.add_trace(go.Scatter(x=cdf["date"], y=cdf["death_acceleration"] * 100,
                                 mode="lines", name="Death Accel. %/wk",
                                 line=dict(color="#fbbf24", width=1.5, dash="dot"),
                                 yaxis="y2"))
    fig.update_layout(**PL,
        title=dict(text="Daily Deaths + Death Acceleration Signal", font=dict(size=14, color="#e2e8f0")),
        barmode="overlay",
        yaxis2=dict(title="%/week", overlaying="y", side="right",
                    gridcolor="#1e2d4a", zerolinecolor="#1e2d4a", showgrid=False))
    st.plotly_chart(fig, use_container_width=True)

# ── SECTION: Case-Death Lag Analysis ─────────────────────────────────────────
if "death_case_lag_corr" in cdf.columns:
    st.markdown('<div class="section-header">Case-to-Death Lag Validation</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cdf["date"], y=cdf["death_case_lag_corr"],
                             mode="lines", name="Rolling Correlation (21d)",
                             line=dict(color="#34d399", width=2),
                             fill="tozeroy", fillcolor="rgba(52,211,153,0.08)"))
    fig.add_hline(y=0.7, line_dash="dot", line_color="#64748b",
                  annotation_text="Strong lag signal (r=0.7)", annotation_font_color="#64748b")
    fig.update_layout(**PL,
        title=dict(text="Rolling Pearson r: Deaths[t] vs Cases[t−14d] — 21-day window",
                   font=dict(size=14, color="#e2e8f0")),
        yaxis=dict(**PL["yaxis"], range=[-1, 1]))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "A high positive correlation confirms the 14-day death lag assumption for this country. "
        "When r drops, it may indicate changes in surveillance quality, CFR, or variant behaviour."
    )

# ── SECTION: Forecast ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">14-Day Forecast</div>', unsafe_allow_html=True)

forecast_df = forecast_cases(df, country)
if not forecast_df.empty:
    hist = cdf[cdf["date"] >= cdf["date"].max() - pd.Timedelta(days=60)]
    fig  = go.Figure()
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["cases_smooth"],
                             mode="lines", name="Historical (smoothed)",
                             line=dict(color="#60a5fa", width=2)))
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_df["date"], forecast_df["date"][::-1]]),
        y=pd.concat([forecast_df["upper_80"], forecast_df["lower_80"][::-1]]),
        fill="toself", fillcolor="rgba(139,92,246,0.12)",
        line=dict(color="rgba(0,0,0,0)"), name="80% CI", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=forecast_df["date"], y=forecast_df["predicted"],
                             mode="lines", name="Forecast",
                             line=dict(color="#8b5cf6", width=2.5, dash="dot")))
    fig.update_layout(**PL,
        title=dict(text="Damped Exponential Forecast with 80% Uncertainty Bands",
                   font=dict(size=14, color="#e2e8f0")))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Day 14 central estimate: ~{int(forecast_df['predicted'].iloc[-1]):,} smoothed cases  |  "
        f"80% CI: [{int(forecast_df['lower_80'].iloc[-1]):,} – {int(forecast_df['upper_80'].iloc[-1]):,}]  |  "
        "Assumes no major new interventions."
    )
else:
    st.info("Insufficient data for forecast.")

# ── SECTION: Global Risk Map ──────────────────────────────────────────────────
st.markdown('<div class="section-header">Global Risk Map</div>', unsafe_allow_html=True)

latest_global = df.sort_values("date").groupby("country").tail(1)
fig = px.choropleth(
    latest_global,
    locations="country", locationmode="country names",
    color="risk_score",
    color_continuous_scale=[[0,"#0f1629"],[0.25,"#1e3a5f"],[0.5,"#1d4ed8"],[0.75,"#f97316"],[1,"#ef4444"]],
    range_color=(0, 1),
    hover_data={"Rt": ":.2f", "growth_rate": ":.3f", "risk_score": ":.3f",
                "transmission_momentum": ":.3f"},
    labels={"risk_score": "Risk", "growth_rate": "Growth"},
)
fig.update_layout(**PL,
    geo=dict(bgcolor="#0a0e1a", lakecolor="#0a0e1a", landcolor="#0f1629",
             showframe=False, showcoastlines=True, coastlinecolor="#1e2d4a"),
    coloraxis_colorbar=dict(title="Risk", tickfont=dict(color="#94a3b8"),
                            titlefont=dict(color="#94a3b8")),
    height=500)
st.plotly_chart(fig, use_container_width=True)

# ── SECTION: Leaderboard ──────────────────────────────────────────────────────
st.markdown('<div class="section-header">High-Risk Country Leaderboard</div>', unsafe_allow_html=True)

display_cols = ["country", "risk_score", "Rt", "growth_rate", "CFR",
                "doubling_time", "transmission_momentum", "healthcare_pressure_norm", "new_cases"]
available = [c for c in display_cols if c in latest_global.columns]
top10 = latest_global.sort_values("risk_score", ascending=False).head(10)[available].copy()
top10["Risk Level"] = top10["risk_score"].apply(classify_risk)
if "growth_rate" in top10.columns:
    top10["growth_rate"] = (top10["growth_rate"] * 100).round(1).astype(str) + "%"
if "CFR" in top10.columns:
    top10["CFR"] = (top10["CFR"] * 100).round(2).astype(str) + "%"
if "doubling_time" in top10.columns:
    top10["doubling_time"] = top10["doubling_time"].apply(lambda x: f"{x:.1f}d" if x > 0 else "—")
if "healthcare_pressure_norm" in top10.columns:
    top10["healthcare_pressure_norm"] = (top10["healthcare_pressure_norm"] * 100).round(0).astype(str) + "%"
if "new_cases" in top10.columns:
    top10["new_cases"] = top10["new_cases"].apply(lambda x: f"{int(x):,}")
for col in ["risk_score", "Rt", "transmission_momentum"]:
    if col in top10.columns:
        top10[col] = top10[col].round(3)

top10.columns = [c.replace("_", " ").title() for c in top10.columns]
st.dataframe(top10.reset_index(drop=True), use_container_width=True, hide_index=True)

# ── SECTION: Cross-Country Rt Comparison ─────────────────────────────────────
st.markdown('<div class="section-header">Cross-Country Rt Comparison (Top 10)</div>', unsafe_allow_html=True)

top_countries = latest_global.sort_values("risk_score", ascending=False).head(10)["country"].tolist()
compare_df    = df[df["country"].isin(top_countries)].copy()
palette = ["#60a5fa","#f97316","#22c55e","#8b5cf6","#f43f5e",
           "#06b6d4","#fbbf24","#a78bfa","#34d399","#fb923c"]

fig = go.Figure()
for i, c in enumerate(top_countries):
    cdata = compare_df[compare_df["country"] == c]
    cdata = cdata[cdata["date"] >= cdata["date"].max() - pd.Timedelta(days=180)]
    fig.add_trace(go.Scatter(x=cdata["date"], y=cdata["Rt"], mode="lines", name=c,
                             line=dict(color=palette[i % len(palette)], width=1.5)))
fig.add_hline(y=1.0, line_dash="dash", line_color="#475569", annotation_text="Rt = 1")
fig.update_layout(**PL,
    title=dict(text="Rt Trajectory — Top 10 Risk Countries (Last 180 days)",
               font=dict(size=14, color="#e2e8f0")),
    yaxis=dict(**PL["yaxis"], range=[0, 4]), height=400)
st.plotly_chart(fig, use_container_width=True)

# ── SECTION: Hotspot Detection ────────────────────────────────────────────────
st.markdown('<div class="section-header">Hotspot Detection</div>', unsafe_allow_html=True)

if hotspot_df is not None and not hotspot_df.empty:
    h = hotspot_df.copy()
    n_hotspots = int(h["is_hotspot"].sum()) if "is_hotspot" in h.columns else 0
    hc1, hc2 = st.columns([2, 3])

    with hc1:
        st.metric("Active Hotspots", n_hotspots,
                  help="Countries with Rt > 1 AND growth rate in global top 25%")
        fig = go.Figure(go.Bar(
            x=h["hotspot_score"].head(15), y=h["country"].head(15),
            orientation="h",
            marker=dict(color=h["hotspot_score"].head(15),
                        colorscale=[[0,"#1e3a5f"],[0.5,"#f97316"],[1,"#ef4444"]],
                        showscale=False)))
        fig.update_layout(**PL,
            title=dict(text="Hotspot Score — Top 15", font=dict(size=13, color="#e2e8f0")),
            yaxis=dict(**PL["yaxis"], autorange="reversed"), height=380)
        st.plotly_chart(fig, use_container_width=True)

    with hc2:
        hcols = ["country", "hotspot_score", "Rt", "growth_rate", "risk_score"]
        hav   = [c for c in hcols if c in h.columns]
        hd    = h[hav].head(15).copy()
        if "growth_rate" in hd.columns:
            hd["growth_rate"] = (hd["growth_rate"] * 100).round(1).astype(str) + "%"
        for col in ["hotspot_score", "Rt", "risk_score"]:
            if col in hd.columns:
                hd[col] = hd[col].round(3)
        hd.columns = [c.replace("_", " ").title() for c in hd.columns]
        st.dataframe(hd.reset_index(drop=True), use_container_width=True, hide_index=True)
else:
    st.info("Run pipeline to generate hotspot data.")

# ── SECTION: Mobility & Transmission ─────────────────────────────────────────
st.markdown('<div class="section-header">Mobility & Transmission Analysis</div>', unsafe_allow_html=True)

if mob_df is not None:
    mob_country = mob_df[mob_df["country"] == country].copy()
    if not mob_country.empty:
        mc1, mc2 = st.columns(2)

        with mc1:
            fig = go.Figure()
            mob_channels = {
                "retail_and_recreation_percent_change_from_baseline": ("Retail & Recreation", "#60a5fa"),
                "transit_stations_percent_change_from_baseline":      ("Transit",             "#f97316"),
                "workplaces_percent_change_from_baseline":            ("Workplaces",          "#22c55e"),
                "residential_percent_change_from_baseline":           ("Residential",         "#f43f5e"),
            }
            for col, (label, color) in mob_channels.items():
                if col in mob_country.columns:
                    fig.add_trace(go.Scatter(x=mob_country["date"], y=mob_country[col],
                                             mode="lines", name=label,
                                             line=dict(color=color, width=1.5)))
            fig.add_hline(y=0, line_dash="dash", line_color="#475569",
                          annotation_text="Baseline (Jan–Feb 2020)")
            fig.update_layout(**PL,
                title=dict(text=f"Mobility Trends — {country}", font=dict(size=14, color="#e2e8f0")),
                height=380)
            st.plotly_chart(fig, use_container_width=True)

        with mc2:
            if "contact_index_lag14" in cdf.columns and cdf["contact_index_lag14"].notna().sum() > 10:
                valid = cdf.dropna(subset=["contact_index_lag14"])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=valid["date"], y=valid["contact_index_lag14"],
                                         mode="lines", name="Contact Index (lag 14d)",
                                         line=dict(color="#06b6d4", width=2), yaxis="y"))
                fig.add_trace(go.Scatter(x=valid["date"], y=valid["Rt"],
                                         mode="lines", name="Effective Rt",
                                         line=dict(color="#f97316", width=2), yaxis="y2"))
                fig.update_layout(**PL,
                    title=dict(text="Contact Index (lag 14d) vs Rt", font=dict(size=14, color="#e2e8f0")),
                    yaxis=dict(**PL["yaxis"], title="Contact Index (%)"),
                    yaxis2=dict(title="Rt", overlaying="y", side="right",
                                gridcolor="#1e2d4a", zerolinecolor="#1e2d4a"),
                    height=380)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Mobility-Rt overlay requires pipeline to be run with mobility enabled.")

        # Mobility-Rt correlation bar chart
        if "contact_index_lag14" in df.columns:
            corr_df = compute_mobility_rt_correlation(df)
            if not corr_df.empty:
                fig = go.Figure(go.Bar(
                    x=corr_df.head(20)["country"],
                    y=corr_df.head(20)["mobility_rt_correlation"],
                    marker=dict(color=corr_df.head(20)["mobility_rt_correlation"],
                                colorscale=[[0,"#22c55e"],[0.5,"#eab308"],[1,"#ef4444"]],
                                showscale=False)))
                fig.add_hline(y=0, line_color="#475569", line_dash="dash")
                fig.update_layout(**PL,
                    title=dict(text="Pearson r: Mobility → Rt (14d lag, top 20 countries)",
                               font=dict(size=13, color="#e2e8f0")),
                    xaxis=dict(**PL["xaxis"], tickangle=-30), height=300)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Positive r = mobility leads transmission. High r countries benefit most from mobility-based interventions.")
    else:
        st.info(f"No mobility data available for {country}.")
else:
    st.info("Run pipeline with `include_mobility=True` to enable mobility analysis.")


# ── SECTION: Vaccination & Testing Context ────────────────────────────────
st.markdown('<div class="section-header">Vaccination & Testing Context</div>', unsafe_allow_html=True)

has_vacc  = "vacc_fully" in cdf.columns and cdf["vacc_fully"].max() > 0
has_test  = "test_positivity_rate" in cdf.columns and cdf["test_positivity_rate"].max() > 0

if has_vacc or has_test:
    vt1, vt2 = st.columns(2)

    with vt1:
        if has_vacc:
            fig = go.Figure()
            if "vacc_one_dose" in cdf.columns:
                fig.add_trace(go.Scatter(x=cdf["date"], y=cdf["vacc_one_dose"],
                                         mode="lines", name="At least one dose",
                                         line=dict(color="#60a5fa", width=2)))
            fig.add_trace(go.Scatter(x=cdf["date"], y=cdf["vacc_fully"],
                                     mode="lines", name="Fully vaccinated",
                                     line=dict(color="#22c55e", width=2.5),
                                     fill="tozeroy", fillcolor="rgba(34,197,94,0.08)"))
            fig.add_hline(y=70, line_dash="dot", line_color="#475569",
                          annotation_text="70% herd threshold (indicative)",
                          annotation_font_color="#64748b")
            fig.update_layout(**PL,
                title=dict(text=f"Vaccination Coverage — {country} (% population)",
                           font=dict(size=14, color="#e2e8f0")),
                yaxis=dict(**PL["yaxis"], range=[0, 105]))
            st.plotly_chart(fig, use_container_width=True)
            # Latest coverage metric
            latest_vacc = cdf["vacc_fully"].replace(0, np.nan).dropna()
            if not latest_vacc.empty:
                st.caption(f"Latest full coverage: **{latest_vacc.iloc[-1]:.1f}%**")
        else:
            st.info("Vaccination data not available for this country in the OWID dataset.")

    with vt2:
        if has_test:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cdf["date"], y=cdf["test_positivity_rate"] * 100,
                                     mode="lines", name="Test Positivity %",
                                     line=dict(color="#f97316", width=2),
                                     fill="tozeroy", fillcolor="rgba(249,115,22,0.08)"))
            # WHO 5% threshold line
            fig.add_hline(y=5, line_dash="dash", line_color="#eab308",
                          annotation_text="WHO 5% threshold — above this, cases are undercounted",
                          annotation_font_color="#eab308")
            fig.add_hline(y=20, line_dash="dot", line_color="#ef4444",
                          annotation_text="20% — severe underdetection",
                          annotation_font_color="#ef4444")

            if "tests_per_thousand" in cdf.columns and cdf["tests_per_thousand"].max() > 0:
                fig.add_trace(go.Scatter(x=cdf["date"], y=cdf["tests_per_thousand"],
                                         mode="lines", name="Tests/1000 pop",
                                         line=dict(color="#06b6d4", width=1.5, dash="dot"),
                                         yaxis="y2"))
                fig.update_layout(**PL,
                    title=dict(text=f"Test Positivity Rate & Testing Volume — {country}",
                               font=dict(size=14, color="#e2e8f0")),
                    yaxis=dict(**PL["yaxis"], title="Positivity %"),
                    yaxis2=dict(title="Tests/1000", overlaying="y", side="right",
                                gridcolor="#1e2d4a", zerolinecolor="#1e2d4a", showgrid=False))
            else:
                fig.update_layout(**PL,
                    title=dict(text=f"Test Positivity Rate — {country}",
                               font=dict(size=14, color="#e2e8f0")))

            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Above the WHO 5% threshold, reported case counts underestimate true spread. "
                "High positivity + rising Rt is a more alarming combination than either alone."
            )
        else:
            st.info("Testing data not available for this country in the OWID dataset.")

    # Vaccination vs Rt overlay — does higher coverage correlate with lower Rt?
    if has_vacc and "Rt" in cdf.columns:
        valid = cdf[cdf["vacc_fully"] > 0].copy()
        if len(valid) > 30:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=valid["date"], y=valid["vacc_fully"],
                                     mode="lines", name="Fully Vaccinated %",
                                     line=dict(color="#22c55e", width=2), yaxis="y"))
            fig.add_trace(go.Scatter(x=valid["date"], y=valid["Rt"],
                                     mode="lines", name="Effective Rt",
                                     line=dict(color="#f97316", width=2), yaxis="y2"))
            fig.add_hline(y=1.0, line_dash="dash", line_color="#475569",
                          annotation_text="Rt=1", yref="y2")
            fig.update_layout(**PL,
                title=dict(text="Vaccination Coverage vs Rt — Context for Transmission",
                           font=dict(size=14, color="#e2e8f0")),
                yaxis=dict(**PL["yaxis"], title="Fully Vaccinated %", range=[0, 105]),
                yaxis2=dict(title="Rt", overlaying="y", side="right",
                            gridcolor="#1e2d4a", zerolinecolor="#1e2d4a",
                            range=[0, max(float(valid["Rt"].max()) + 0.5, 3)]),
                height=350)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "When Rt rises despite high vaccination coverage, it signals waning immunity "
                "or an immune-evasive variant — a different risk profile than an unvaccinated population."
            )
else:
    st.info(
        "Vaccination and testing data require the OWID source. "
        "Ensure the pipeline ran successfully with OWID access."
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#374151;font-size:0.8rem'>"
    "EpiIQ Sentinel · JHU CSSE · Google Mobility · "
    "Rt ratio method (SI=7d) · CFR lagged 14d · "
    "CUSUM wave detection · Damped exponential forecast (80% CI)"
    "</div>",
    unsafe_allow_html=True,
)

