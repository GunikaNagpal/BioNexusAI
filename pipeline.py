import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
DATA_DIR.mkdir(parents=True, exist_ok=True)

JHU_BASE = (
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
    "csse_covid_19_data/csse_covid_19_time_series/"
)

def load_jhu():
    confirmed = pd.read_csv(JHU_BASE + "time_series_covid19_confirmed_global.csv")
    deaths = pd.read_csv(JHU_BASE + "time_series_covid19_deaths_global.csv")

    def reshape(df, value_name):
        df = df.drop(columns=["Province/State", "Lat", "Long"])
        df = df.groupby("Country/Region").sum()
        df = df.T
        df.index = pd.to_datetime(df.index)
        df = df.reset_index().melt(id_vars="index", var_name="country", value_name=value_name)
        df.rename(columns={"index": "date"}, inplace=True)
        return df

    confirmed = reshape(confirmed, "total_cases")
    deaths = reshape(deaths, "total_deaths")

    df = confirmed.merge(deaths, on=["country", "date"])
    return df

def compute_features(df):
    df = df.sort_values(["country", "date"])
    grp = df.groupby("country")

    df["new_cases"] = grp["total_cases"].diff().clip(lower=0)
    df["new_deaths"] = grp["total_deaths"].diff().clip(lower=0)

    df["cases_smooth"] = grp["new_cases"].transform(lambda x: x.rolling(7).mean())
    df["growth_rate"] = grp["cases_smooth"].pct_change()

    df["Rt"] = grp["cases_smooth"].transform(
        lambda x: (x + 1) / (x.shift(7) + 1)
    )

    df["CFR"] = df["total_deaths"] / (df["total_cases"] + 1)

    df = df.fillna(0)

    # Simple interpretable risk score
    df["risk_score"] = (
        0.4 * df["growth_rate"] +
        0.3 * df["Rt"] +
        0.3 * df["CFR"]
    ).clip(0, 2)

    return df

def run_pipeline():
    print("Loading JHU data...")
    df = load_jhu()

    print("Computing features...")
    df = compute_features(df)

    # 🔥 REDUCE DATA SIZE (IMPORTANT FOR DEPLOYMENT)
    df = df[df["date"] >= "2022-01-01"]

    print("Saving processed data...")
    df.to_csv(DATA_DIR / "risk.csv", index=False)

    print("Pipeline complete. File ready for deployment.")

if __name__ == "__main__":
    run_pipeline()
