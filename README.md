# BioNexusAI - **EpiIQ-Sentinel**
AI-Powered Spatio-Temporal Epidemic Intelligence System | CodeCure Hackathon | IIT BHU
#  EpiIQ Sentinel - See the Outbreak before it sees you!!

> **AI-Powered Spatio-Temporal Epidemic Intelligence & Early Warning System**


**Team BioNexusAI** · CodeCure AI Hackathon · SPIRIT'26 · IIT BHU Varanasi
**Track C:** Epidemic Spread Prediction (Epidemiology + AI)

> **Epi** — Epidemiological · **IQ** — AI-driven Intelligence · **Sentinel** — Early Warning Surveillance


# The Problem

Most epidemic dashboards show you a number like cases, deaths, a map coloured by severity. You see a red country. **So what?**

They don't tell you *why* it's red, whether neighbours are at risk, what happens if vaccination drops, or where the outbreak will be in two weeks.

**EpiIQ Sentinel** answers the four questions that actually matter to public health decision-makers:


| Question | Module |

| 📍 **WHERE** is the outbreak — is it spreading across borders? | Spatial analysis |

| ⏱️ **WHEN** did the outbreak change and why? | Structural break detection + variant attribution |

| 📈 **WHAT** will happen in the next 14 days? | ARIMA / Prophet forecasting |

| 🧠 **WHY** is this country high risk? | SHAP explainability in biological language |



EpiIQ-Sentinel/

│

├── app.py

├── ingest.py

├── preprocess.py

├── outbreak.py

├── spatial.py

├── structural.py

├── forecast.py

├── risk.py

│

├── requirements.txt

├── README.md



# Architecture


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



EpiIQ Sentinel is an AI-powered spatio-temporal epidemic intelligence system designed to transform raw epidemic data into actionable public health insights. Unlike traditional dashboards that only display case counts or deaths, this system answers four critical questions: where an outbreak is spreading, when and why its dynamics changed, what is likely to happen next, and why a region is considered high risk. The system begins by ingesting global COVID-19 data from sources like Our World in Data, which is then preprocessed to generate key epidemiological indicators such as smoothed case trends, growth rate, and a lag-adjusted case fatality ratio. It then performs outbreak analysis by estimating the effective reproduction number (Rt) to measure transmission intensity and detecting waves based on sudden increases in growth rate. To understand spatial spread, the system models global and regional transmission pressure using WHO regions, allowing it to approximate how outbreaks influence neighboring regions. It further identifies structural breaks by detecting abrupt changes in growth patterns and associates these shifts with possible causes such as new variants or policy interventions. For future planning, the system uses time-series models like Prophet and ARIMA to forecast cases for the next 14 days. A machine learning model (XGBoost) then classifies regions into high or low risk based on epidemiological features, while explainable AI techniques like SHAP provide clear, human-readable reasons behind the risk predictions. Additionally, a scenario simulation module allows users to explore how changes in mobility or vaccination levels could impact future cases. All these components are integrated into an interactive Streamlit dashboard that includes a global risk map, animated outbreak timeline, hotspot detection, epidemiological signal panels, forecasts, AI explanations, alert systems, and policy simulations. Overall, EpiIQ Sentinel moves beyond static visualization and acts as a comprehensive decision-support system, enabling policymakers to anticipate, understand, and respond to epidemic threats more effectively.




