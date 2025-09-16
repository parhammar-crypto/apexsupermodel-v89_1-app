
import streamlit as st
import subprocess
import json
import datetime

st.set_page_config(page_title="ApexSuperModel V89.1", layout="centered")
st.title("⚽ ApexSuperModel V89.1 – Backtest & Prediction")

mode = st.selectbox("Välj läge", ["predict", "backtest"])
league = st.selectbox("Välj liga", ["Premier League", "LaLiga", "Serie A", "Bundesliga", "Ligue 1"])
save_output = st.checkbox("Spara output?", value=True)
chat_output = st.checkbox("Visa chat-liknande förklaring?", value=True)

if mode == "predict":
    date_from = st.date_input("Från datum", datetime.date.today())
    date_to = st.date_input("Till datum", datetime.date.today() + datetime.timedelta(days=2))
    if st.button("🔮 Kör prediction"):
        config = {
            "mode": "predict",
            "leagues": [league],
            "date_from": str(date_from),
            "date_to": str(date_to),
            "save": save_output,
            "chat": chat_output
        }
        st.code(json.dumps(config, indent=2), language="json")
        result = subprocess.run(["python", "apex_supermodel_v89_1.py"], input=json.dumps(config), text=True, capture_output=True)
        st.text_area("📤 Output", result.stdout)
        if result.stderr:
            st.error(result.stderr)

elif mode == "backtest":
    train_years = st.multiselect("Träningsår", [2020, 2021, 2022, 2023], default=[2021, 2022, 2023])
    test_year = st.selectbox("Testår", [2023, 2024, 2025])
    if st.button("📊 Kör backtest"):
        config = {
            "mode": "backtest",
            "leagues": [league],
            "train_years": train_years,
            "test_year": test_year,
            "save": save_output,
            "chat": chat_output
        }
        st.code(json.dumps(config, indent=2), language="json")
        result = subprocess.run(["python", "apex_supermodel_v89_1.py"], input=json.dumps(config), text=True, capture_output=True)
        st.text_area("📤 Output", result.stdout)
        if result.stderr:
            st.error(result.stderr)
