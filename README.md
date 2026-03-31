# BioNexusAI
AI-Powered Spatio-Temporal Epidemic Intelligence System | CodeCure Hackathon | IIT BHU
#  EpiIQ Sentinel - See the Outbreak before it sees you!!

> **AI-Powered Spatio-Temporal Epidemic Intelligence & Early Warning System**


**Team BioNexusAI** · CodeCure AI Hackathon · SPIRIT'26 · IIT BHU Varanasi
**Track C:** Epidemic Spread Prediction (Epidemiology + AI)

> **Epi** — Epidemiological · **IQ** — AI-driven Intelligence · **Sentinel** — Early Warning Surveillance


##  The Problem

Most epidemic dashboards show you a number like cases, deaths, a map coloured by severity. You see a red country. **So what?**

They don't tell you *why* it's red, whether neighbours are at risk, what happens if vaccination drops, or where the outbreak will be in two weeks.

**EpiIQ Sentinel** answers the four questions that actually matter to public health decision-makers:

| Question | Module |
| 📍 **WHERE** is the outbreak — is it spreading across borders? | Spatial analysis 
| ⏱️ **WHEN** did the outbreak change and why? | Structural break detection + variant attribution |
| 📈 **WHAT** will happen in the next 14 days? | ARIMA / Prophet forecasting |
| 🧠 **WHY** is this country high risk? | SHAP explainability in biological language |


##  Architecture

Data Sources  (JHU · OWID · Google Mobility)
        ↓
ingest.py        — reshape, ISO3 standardise, merge
        ↓
preprocess.py    — incident counts, smoothing, CFR, growth rate
        ↓
outbreak.py      — Rt, structural breaks, wave detection, early warning
        ↓
spatial.py       — 
        ↓
forecast.py      — ARIMA (endemic) / Prophet (acute) · 14-day forecast
        ↓
risk.py          — 6-dim risk score · XGBoost · Isolation Forest · SHAP
        ↓
app.py           —  Streamlit dashboard


# AI Layers

| Layer | Method | Purpose |
| Statistical AI | ARIMA · Prophet + changepoints | 14-day case forecasting |
| ML AI | XGBoost · Isolation Forest | Risk classification + anomaly detection |
| Explainable AI | SHAP TreeExplainer | Why is this country high risk? |
| Scenario AI | What-if sliders | Policy simulation (vaccination, mobility) |


# Datasets

| Source | Description | Link |
| JHU CSSE | Daily confirmed, deaths, recovered | [GitHub](https://github.com/CSSEGISandData/COVID-19) |
| Our World in Data | Vaccination, testing, demographics | [GitHub](https://github.com/owid/covid-19-data) |
| Google Mobility | Movement trends by category | [Google](https://www.google.com/covid19/mobility/) |






