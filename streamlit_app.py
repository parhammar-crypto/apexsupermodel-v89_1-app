# streamlit_app.py
import streamlit as st
import pandas as pd
from apex_supermodel_v89_1 import (
    ApexSuperModelV891, prep_base, add_venue_form, add_market_drift,
    QuantumDataManager, TRAIN_YEARS, TEST_YEAR, parse_fixtures
)

st.set_page_config(page_title="ApexSuperModel V89.1 – Backtest & Prediction", layout="wide")

st.title("⚽ ApexSuperModel V89.1 – Backtest & Prediction")

mode = st.selectbox("Välj läge", ["backtest", "predict"])
league = st.selectbox("Välj liga", [
    "Premier League","LaLiga","Bundesliga","Serie A","Ligue 1",
    "Eredivisie","Primeira Liga","Jupiler Pro League"
], index=0)
save = st.checkbox("Spara output?", value=False)
chat = st.checkbox("Visa chat-liknande förklaring?", value=True)

train_years = st.multiselect("Träningsår", [2020,2021,2022,2023], default=[2021,2022,2023])
test_year = st.selectbox("Testår", [2024], index=0)

st.code({
    "mode": mode,
    "leagues": [league],
    "train_years": train_years,
    "test_year": test_year,
    "save": save,
    "chat": chat
}, language="json")

if st.button("🚀 Kör"):
    dm = QuantumDataManager("./apex_cache_super_v89_1", timeout=15.0, rate_limit=1.6)
    st.write("🔄 Laddar historik…")
    hist = pd.concat([dm.fetch_season(league, y) for y in train_years + [test_year]], ignore_index=True)
    if hist.empty:
        st.error("Ingen historik hittades – kontrollera liga/år.")
        st.stop()

    data = prep_base(hist)
    data = add_venue_form(data)
    data = add_market_drift(data)

    model = ApexSuperModelV891(conformal_level=0.85)
    st.write("🧠 Tränar modellen…")
    model.fit(data)

    # Backtest: använd teståret
    test_df = data[data["Season"] == test_year].copy()
    if test_df.empty:
        st.warning("Inga matcher i teståret – visar träningsårens sista säsong i stället.")
        test_df = data[data["Season"] == max(train_years)].copy()

    st.write("📈 Skapar prediktioner…")
    preds = model.predict_df(test_df.dropna(subset=["Home","Away","League"]))
    st.dataframe(preds.head(50))

    if save:
        preds.to_csv("preds_v89_1.csv", index=False)
        st.success("✅ Sparade preds_v89_1.csv")
