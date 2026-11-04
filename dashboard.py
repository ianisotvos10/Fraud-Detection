import json
import os

import pandas as pd
import streamlit as st

from src.config import REPORTS_DIR, THRESHOLD_PATH

st.set_page_config(page_title="Monitorizare model frauda", layout="wide")
st.title("Monitorizare model de detectie a fraudei")


def list_scenarios():
    if not os.path.isdir(REPORTS_DIR):
        return []
    names = []
    for name in os.listdir(REPORTS_DIR):
        if name.startswith("drift_report_") and name.endswith(".csv"):
            names.append(name[len("drift_report_"):-len(".csv")])
    return sorted(names)


scenarios = list_scenarios()
if not scenarios:
    st.warning("Nu exista rapoarte. Ruleaza: python -m src.monitor --scenario natural")
    st.stop()

scenario = st.sidebar.selectbox("Scenariu", scenarios)
report = pd.read_csv(os.path.join(REPORTS_DIR, f"drift_report_{scenario}.csv"))
alerted = report.loc[report["alert"], "batch"].tolist()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Loturi analizate", len(report))
col2.metric("Loturi cu alerta", len(alerted))
col3.metric("PSI maxim scor", f"{report['score_psi'].max():.3f}")
if os.path.exists(THRESHOLD_PATH):
    with open(THRESHOLD_PATH) as f:
        col4.metric("Prag model", f"{json.load(f)['threshold']:.4f}")

if alerted:
    st.error(f"Drift detectat in loturile: {alerted}")
else:
    st.success("Nu s-a detectat drift in niciun lot.")

st.subheader("Drift (PSI)")
st.line_chart(report.set_index("batch")[["score_psi", "max_feature_psi"]])
st.caption("Pragul de alerta este PSI = 0.2")

left, right = st.columns(2)
with left:
    st.subheader("Performanta pe loturi")
    st.line_chart(report.set_index("batch")[["recall", "precision"]])
    fraude = ", ".join(f"lot {int(b)}: {int(n)}" for b, n in zip(report["batch"], report["frauds"]))
    st.caption(f"Numar de fraude per lot: {fraude}. Valorile pe putine fraude sunt zgomotoase.")
with right:
    if "flag_rate" in report.columns:
        st.subheader("Procent tranzactii marcate ca frauda")
        st.line_chart((report.set_index("batch")[["flag_rate"]] * 100))

st.subheader("Raport detaliat")
st.dataframe(report, use_container_width=True)

image_path = os.path.join(REPORTS_DIR, f"drift_{scenario}.png")
if os.path.exists(image_path):
    st.subheader("Grafic salvat")
    st.image(image_path)
