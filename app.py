import streamlit as st
import numpy as np
import pickle
import shap
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Healthcare Disease Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #080E14;
    color: #C8D8E8;
}

.stApp {
    background: #080E14;
}

section[data-testid="stSidebar"] {
    background: #0D1520;
    border-right: 1px solid #1A2D42;
    min-width: 260px !important;
    width: 260px !important;
}

section[data-testid="stSidebar"] * {
    color: #8AACCC !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
}

section[data-testid="stSidebar"] .stNumberInput input {
    background: #0A1929 !important;
    border: 1px solid #1A3A5C !important;
    border-radius: 4px !important;
    color: #00E5FF !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 14px !important;
    padding: 8px 12px !important;
}

section[data-testid="stSidebar"] label {
    color: #4A7FA8 !important;
    font-size: 11px !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}

div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #003D66 0%, #00264D 100%) !important;
    border: 1px solid #00AACC !important;
    border-radius: 3px !important;
    color: #00E5FF !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    letter-spacing: 0.15em !important;
    padding: 12px 32px !important;
    text-transform: uppercase !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 0 20px rgba(0, 200, 255, 0.15) !important;
}

div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #005580 0%, #003866 100%) !important;
    box-shadow: 0 0 30px rgba(0, 200, 255, 0.3) !important;
}

.stDataFrame {
    background: #0D1520 !important;
    border: 1px solid #1A2D42 !important;
    border-radius: 4px !important;
}

.stDataFrame table {
    background: transparent !important;
}

.stDataFrame th {
    background: #0A1929 !important;
    color: #4A7FA8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid #1A3A5C !important;
}

.stDataFrame td {
    color: #C8D8E8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
    border-bottom: 1px solid #0D1520 !important;
}

.stAlert {
    background: #0A1929 !important;
    border-left: 3px solid #00AACC !important;
    border-radius: 3px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
}

.block-container {
    padding: 1.5rem 2rem 4rem !important;
    max-width: 100% !important;
}

[data-testid="stAppViewContainer"] > section:last-child {
    max-width: 100% !important;
}

.stMetric {
    background: #0D1520 !important;
    border: 1px solid #1A2D42 !important;
    border-radius: 4px !important;
    padding: 1rem !important;
}

.stMetric label {
    color: #4A7FA8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}

.stMetric [data-testid="stMetricValue"] {
    color: #00E5FF !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 28px !important;
}

[data-testid="stMarkdownContainer"] p {
    font-family: 'JetBrains Mono', monospace;
    color: #8AACCC;
    font-size: 13px;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #080E14; }
::-webkit-scrollbar-thumb { background: #1A3A5C; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
    border-bottom: 1px solid #1A2D42;
    padding: 0 0 1.5rem 0;
    margin-bottom: 2rem;
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
">
    <div>
        <div style="
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            letter-spacing: 0.2em;
            color: #00AACC;
            text-transform: uppercase;
            margin-bottom: 6px;
        ">Healthcare-Disease-Prediction — v1.0 — Diagnostic System</div>
        <h1 style="
            font-family: 'Syne', sans-serif;
            font-size: 36px;
            font-weight: 800;
            color: #E8F4FF;
            margin: 0;
            letter-spacing: -0.02em;
        ">Healthcare <span style='color: #00AACC;'>Disease</span> Prediction</h1>
    </div>
    <div style="
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: #1A4A6A;
        text-align: right;
        line-height: 2;
    ">
        <div>● SYSTEM ONLINE</div>
        <div>MODEL: XGBoost v1.7</div>
        <div>DATASET: PIMA Indian Diabetes</div>
    </div>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return pickle.load(open("models/model.pkl", "rb"))

model = load_model()

st.sidebar.markdown("""
<div style="
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.2em;
    color: #2A5A8A;
    text-transform: uppercase;
    border-bottom: 1px solid #1A2D42;
    padding-bottom: 12px;
    margin-bottom: 16px;
">Patient Biometrics Input</div>
""", unsafe_allow_html=True)

preg     = st.sidebar.number_input("Pregnancies",              0,   20,   1)
glucose  = st.sidebar.number_input("Glucose (mg/dL)",          0,  200, 120)
bp       = st.sidebar.number_input("Blood Pressure (mmHg)",    0,  150,  70)
skin     = st.sidebar.number_input("Skin Thickness (mm)",      0,  100,  20)
insulin  = st.sidebar.number_input("Insulin (μU/mL)",          0,  900,  80)
bmi      = st.sidebar.number_input("BMI",                    0.0, 70.0, 25.0)
dpf      = st.sidebar.number_input("Diabetes Pedigree Fn",   0.0,  2.5,  0.5)
age      = st.sidebar.number_input("Age (years)",              1,  120,  30)

st.sidebar.markdown("---")
run = st.sidebar.button("▶  RUN ANALYSIS")

input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])

FEATURES = ["Pregnancies","Glucose","Blood Pressure",
            "Skin Thickness","Insulin","BMI","DPF","Age"]

REF = {
    "Pregnancies":    (0, 20, preg),
    "Glucose":        (70, 140, glucose),
    "Blood Pressure": (60, 120, bp),
    "Skin Thickness": (0, 60, skin),
    "Insulin":        (0, 200, insulin),
    "BMI":            (18.5, 35, bmi),
    "DPF":            (0, 2.5, dpf),
    "Age":            (1, 100, age),
}

st.markdown("""
<div style="
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.2em;
    color: #2A5A8A;
    text-transform: uppercase;
    margin-bottom: 12px;
">Current Patient Vitals</div>
""", unsafe_allow_html=True)

cols_r1 = st.columns(4)
cols_r2 = st.columns(4)
vital_labels = ["PREG", "GLUC", "BP", "SKIN", "INS", "BMI", "DPF", "AGE"]
vital_vals   = [preg, glucose, bp, skin, insulin, bmi, dpf, age]
vital_units  = ["", " mg/dL", " mmHg", " mm", " μU/mL", "", "", " yr"]

VITAL_RANGES = [
    (0, 10),      # PREG normal
    (70, 140),    # GLUC normal
    (60, 120),    # BP normal
    (0, 50),      # SKIN
    (0, 166),     # INS normal
    (18.5, 30),   # BMI normal
    (0, 1.0),     # DPF normal
    (0, 60),      # AGE
]

def vital_status(val, lo, hi):
    if val > hi * 1.15:
        return "#FF3B3B", "#1A0000", "#FF3B3B40", "high"
    elif val > hi:
        return "#FFB800", "#1A1200", "#FFB80040", "elevated"
    return "#00E5FF", "#0D1520", "#1A2D42", "normal"

all_cols = cols_r1 + cols_r2
for i, col in enumerate(all_cols):
    vc, bg, bc, status = vital_status(vital_vals[i], *VITAL_RANGES[i])
    with col:
        st.markdown(f"""
        <div style="background:{bg};border:1px solid {bc};border-radius:4px;padding:8px 10px;text-align:center;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:0.12em;color:#2A5A8A;text-transform:uppercase;margin-bottom:4px">{vital_labels[i]}</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:16px;color:{vc};font-weight:700">{vital_vals[i]}{vital_units[i]}</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:8px;color:{vc}80;margin-top:2px">{status}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:1rem'></div>", unsafe_allow_html=True)

def make_radar(input_row):
    normed = []
    for f in FEATURES:
        lo, hi, val = REF[f]
        normed.append(round(min(max((val - lo) / (hi - lo), 0), 1), 3))

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=normed + [normed[0]],
        theta=FEATURES + [FEATURES[0]],
        fill="toself",
        fillcolor="rgba(0, 170, 204, 0.12)",
        line=dict(color="#00AACC", width=2),
        name="Patient",
    ))
    fig.add_trace(go.Scatterpolar(
        r=[0.5]*len(FEATURES) + [0.5],
        theta=FEATURES + [FEATURES[0]],
        fill=None,
        line=dict(color="#1A3A5C", width=1, dash="dash"),
        name="Baseline",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#0A1929",
            radialaxis=dict(visible=False, range=[0, 1]),
            angularaxis=dict(
                tickfont=dict(family="JetBrains Mono", size=11, color="#4A7FA8"),
                linecolor="#1A2D42",
                gridcolor="#0D2035",
            )
        ),
        paper_bgcolor="#0D1520",
        plot_bgcolor="#0D1520",
        font=dict(family="JetBrains Mono", color="#8AACCC"),
        legend=dict(
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)",
            bordercolor="#1A2D42",
        ),
        margin=dict(l=50, r=50, t=10, b=10),
        height=260,
    )
    return fig

st.markdown("""
<div style="
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.2em;
    color: #2A5A8A;
    text-transform: uppercase;
    margin-bottom: 8px;
">Biometric Profile — Normalised</div>
""", unsafe_allow_html=True)
st.plotly_chart(make_radar(input_data), use_container_width=True)

if run:
    prob = model.predict_proba(input_data)[0][1]

    if prob < 0.3:
        risk, color, bg, icon = "LOW RISK", "#00FF8C", "#003322", "✔"
        advice = "Biometrics within acceptable ranges. Maintain current lifestyle and schedule annual checkups."
    elif prob < 0.7:
        risk, color, bg, icon = "MODERATE RISK", "#FFB800", "#2A1E00", "⚠"
        advice = "Elevated indicators detected. Recommend dietary adjustments, glucose monitoring, and consultation within 3 months."
    else:
        risk, color, bg, icon = "HIGH RISK", "#FF3B3B", "#2A0000", "✖"
        advice = "Critical thresholds exceeded. Immediate physician consultation strongly recommended."

    st.markdown(f"""
    <div style="
        background: {bg};
        border: 1px solid {color}40;
        border-left: 4px solid {color};
        border-radius: 0;
        padding: 1.5rem 2rem;
        margin: 1.5rem 0 1rem;
        display: flex;
        align-items: center;
        gap: 2rem;
    ">
        <div style="
            font-size: 48px;
            color: {color};
            font-family: 'JetBrains Mono', monospace;
            line-height: 1;
        ">{icon}</div>
        <div>
            <div style="
                font-family: 'JetBrains Mono', monospace;
                font-size: 10px;
                letter-spacing: 0.2em;
                color: {color}99;
                text-transform: uppercase;
                margin-bottom: 4px;
            ">Diagnostic Result</div>
            <div style="
                font-family: 'Syne', sans-serif;
                font-size: 28px;
                font-weight: 800;
                color: {color};
                letter-spacing: 0.05em;
            ">{risk}</div>
            <div style="
                font-family: 'JetBrains Mono', monospace;
                font-size: 13px;
                color: {color}CC;
                margin-top: 4px;
            ">Diabetes Probability: {prob:.1%}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="
        background: #0A1929;
        border: 1px solid #1A2D42;
        border-radius: 3px;
        padding: 1rem 1.25rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        color: #8AACCC;
        margin-bottom: 2rem;
    ">
        <span style='color: #4A7FA8; text-transform: uppercase; font-size: 11px; letter-spacing: 0.1em;'>Clinical Guidance — </span>
        {advice}
    </div>
    """, unsafe_allow_html=True)

    gauge_color = "#00FF8C" if prob < 0.3 else "#FFB800" if prob < 0.7 else "#FF3B3B"
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number=dict(suffix="%", font=dict(family="JetBrains Mono", size=32, color=gauge_color)),
        gauge=dict(
            axis=dict(range=[0, 100], tickfont=dict(family="JetBrains Mono", size=10, color="#4A7FA8")),
            bar=dict(color=gauge_color, thickness=0.3),
            bgcolor="#0A1929",
            borderwidth=0,
            steps=[
                dict(range=[0, 30],  color="#002A1A"),
                dict(range=[30, 70], color="#1A1200"),
                dict(range=[70, 100],color="#1A0000"),
            ],
            threshold=dict(line=dict(color=gauge_color, width=3), thickness=0.8, value=prob * 100),
        )
    ))
    fig_gauge.update_layout(
        paper_bgcolor="#0D1520",
        font=dict(color="#8AACCC"),
        margin=dict(l=20, r=20, t=10, b=10),
        height=200,
    )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)
    sv = shap_values[0] if isinstance(shap_values, list) else shap_values[0]

    shap_df = pd.DataFrame({"Feature": FEATURES, "Impact": sv})
    shap_df["AbsImpact"] = shap_df["Impact"].abs()
    shap_df = shap_df.sort_values("AbsImpact", ascending=True)

    bar_colors = ["#FF3B3B" if v > 0 else "#00AACC" for v in shap_df["Impact"]]

    fig_shap = go.Figure(go.Bar(
        x=shap_df["Impact"],
        y=shap_df["Feature"],
        orientation="h",
        marker=dict(color=bar_colors, line=dict(width=0)),
    ))
    fig_shap.update_layout(
        paper_bgcolor="#0D1520",
        plot_bgcolor="#0A1929",
        font=dict(family="JetBrains Mono", color="#8AACCC", size=12),
        xaxis=dict(
            title="SHAP Value (Risk Impact)",
            gridcolor="#0D2035",
            zerolinecolor="#1A3A5C",
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            gridcolor="#0D2035",
            tickfont=dict(size=11),
        ),
        margin=dict(l=10, r=10, t=10, b=40),
        height=280,
        showlegend=False,
    )
    fig_shap.add_vline(x=0, line_width=1, line_color="#1A3A5C")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:0.2em;color:#2A5A8A;text-transform:uppercase;margin-bottom:8px">
        Risk Probability Gauge</div>""", unsafe_allow_html=True)
        st.plotly_chart(fig_gauge, use_container_width=True)
    with col2:
        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:0.2em;color:#2A5A8A;text-transform:uppercase;margin-bottom:8px">
        SHAP Feature Impact — <span style='color:#FF3B3B'>Red = increases risk</span> · <span style='color:#00AACC'>Blue = reduces risk</span></div>""",
        unsafe_allow_html=True)
        st.plotly_chart(fig_shap, use_container_width=True)

st.markdown("<div style='margin: 2rem 0 0.5rem; border-top: 1px solid #1A2D42; padding-top: 1.5rem;'></div>", unsafe_allow_html=True)

st.markdown("""
<div style="font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:0.2em;color:#2A5A8A;text-transform:uppercase;margin-bottom:12px">
Model Performance Benchmarks</div>
""", unsafe_allow_html=True)

results = pd.DataFrame({
    "Model":    ["Logistic Regression", "Random Forest", "XGBoost"],
    "Accuracy": [0.78, 0.85, 0.88],
    "F1 Score": [0.75, 0.83, 0.87],
    "AUC-ROC":  [0.80, 0.88, 0.91],
})

fig_bench = go.Figure()
metrics = ["Accuracy", "F1 Score", "AUC-ROC"]
bench_colors = ["#4A7FA8", "#00AACC", "#00E5FF"]

for i, m in enumerate(metrics):
    fig_bench.add_trace(go.Bar(
        name=m,
        x=results["Model"],
        y=results[m],
        marker=dict(color=bench_colors[i], opacity=0.85),
        text=[f"{v:.2f}" for v in results[m]],
        textposition="outside",
        textfont=dict(family="JetBrains Mono", size=11, color=bench_colors[i]),
    ))

fig_bench.add_shape(
    type="rect",
    x0=1.5, x1=2.5, y0=0, y1=1,
    fillcolor="rgba(0, 170, 204, 0.05)",
    line=dict(width=0),
    layer="below"
)
fig_bench.add_annotation(
    x=2, y=0.95, text="ACTIVE MODEL",
    font=dict(family="JetBrains Mono", size=9, color="#00AACC"),
    showarrow=False, yref="paper"
)

fig_bench.update_layout(
    barmode="group",
    paper_bgcolor="#0D1520",
    plot_bgcolor="#0A1929",
    font=dict(family="JetBrains Mono", color="#8AACCC", size=12),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02,
        font=dict(size=11),
        bgcolor="rgba(0,0,0,0)",
    ),
    xaxis=dict(gridcolor="#0D2035", tickfont=dict(size=12)),
    yaxis=dict(range=[0, 1.05], gridcolor="#0D2035", tickfont=dict(size=10)),
    margin=dict(l=10, r=10, t=50, b=10),
    height=340,
)
st.plotly_chart(fig_bench, use_container_width=True)

st.markdown("""
<div style="
    border-top: 1px solid #0D1C2E;
    margin-top: 3rem;
    padding-top: 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #1A3A5C;
    display: flex;
    justify-content: space-between;
">
    <span>Healthcare-Disease-Prediction — For Research & Educational Use Only</span>
    <span>Not a substitute for clinical diagnosis</span>
</div>
""", unsafe_allow_html=True)