#  **EpiIQ-Sentinel**
AI-Powered Spatio-Temporal Epidemic Intelligence System | CodeCure Hackathon | IIT BHU
#  EpiIQ Sentinel - See the Outbreak before it sees you!!

> **AI-Powered Spatio-Temporal Epidemic Intelligence & Early Warning System**


**Team BioNexusAI** · CodeCure AI Hackathon · SPIRIT'26 · IIT BHU Varanasi
**Track C:** Epidemic Spread Prediction (Epidemiology + AI)

> **Epi** — Epidemiological · **IQ** — AI-driven Intelligence · **Sentinel** — Early Warning Surveillance


# The Problem

Most epidemic dashboards show you a number like cases, deaths, a map coloured by severity. You see a red country. **So what?**

They don't tell you *why* it's red, whether neighbours are at risk, what happens if vaccination drops, or where the outbreak will be in two weeks.


EpiIQ Sentinel is an AI-powered epidemic intelligence system that transforms raw epidemiological data into actionable early-warning insights.

Unlike traditional dashboards that only display case counts or static maps, EpiIQ Sentinel focuses on prediction, interpretation, and decision support.


**EpiIQ Sentinel** answers the four questions that actually matter to public health decision-makers:


| Question | Module |

| 📍 **WHERE** is the outbreak — is it spreading across borders? | Spatial analysis |

| ⏱️ **WHEN** did the outbreak change and why? | Structural break detection + variant attribution |

| 📈 **WHAT** will happen in the next 14 days? | ARIMA / Prophet forecasting |

| 🧠 **WHY** is this country high risk? | SHAP explainability in biological language |


EpiIQ Sentinel acts as an early warning system by combining epidemiological modeling with data-driven insights.

It processes time-series epidemic data to extract meaningful signals such as:

Effective reproduction number (Rt)

Growth rate and trend dynamics

Case smoothing and noise reduction

Case fatality ratio (CFR)

These signals are integrated to generate a composite risk score, enabling identification of emerging outbreak patterns.

Additionally, the system provides:

📊 Trend visualization

🔮 Short-term forecasting (14-day outlook)

🌍 Global risk mapping

🚨 Early warning alerts based on transmission dynamics

🧠 Human-readable explanations of risk


**Streamlit link** : https://epiq-sentinel-gunikagaurav.streamlit.app/ 



EpiIQ-Sentinel/

│

├── app.py               # Streamlit dashboard

├── ingest.py            # Data loading and merging

├── preprocess.py        # Cleaning and feature engineering

├── outbreak.py          # Epidemiological signals (Rt, growth)

├── spatial.py           # Global risk mapping

├── structural.py        # Trend/shift detection

├── forecast.py          # Time-series forecasting

├── risk.py              # Risk scoring logic

│

├── requirements.txt

├── README.md



# Architecture
        
        ↓
        
Data Sources (JHU / OWID)

        ↓
        
Ingestion → Preprocessing → Signal Extraction

        ↓
        
Outbreak Analysis (Rt, Growth, Trends)

        ↓
        
Forecasting (14-day projection)

        ↓
        
Risk Scoring

        ↓
        
Streamlit Dashboard

📊 Key Features

📈 Epidemiological Signals

Effective reproduction number (Rt)

Growth rate of cases

Smoothed incidence trends

Case fatality ratio (CFR)


🔮 Forecasting

Short-term projection of cases (next 14 days)

Trend-based estimation of future spread

🌍 Spatial Intelligence

Global risk distribution

Identification of high-risk regions


🚨 Early Warning System

Detects potential outbreak phases

Flags regions with rising transmission


🧠 Explainability

Interprets risk using epidemiological indicators

Provides human-readable reasoning behind risk levels


 Tech Stack

Python

Pandas / NumPy

Plotly (visualization)

Streamlit (dashboard)


📂 Data Sources

JHU CSSE COVID-19 Dataset


🎯 Impact

EpiIQ Sentinel shifts epidemic monitoring from:

❌ Passive reporting

➡️

✅ Proactive early warning and interpretation

By combining epidemiological signals with simple predictive modeling, it enables:

Earlier identification of outbreak trends

Better situational awareness

Data-driven decision support


🏁 Conclusion

EpiIQ Sentinel demonstrates how data science + epidemiology can move beyond visualization into intelligence systems.

It provides a foundation for building more advanced tools that support real-world public health decision-making.










