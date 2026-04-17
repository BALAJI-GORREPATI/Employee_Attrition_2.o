import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# =======================================
# PAGE CONFIG
# =======================================
st.set_page_config(page_title="Employee Attrition", layout="wide")

# =======================================
# THEME TOGGLE
# =======================================
theme = st.sidebar.radio("Select Theme", ["Light", "Dark"])

if theme == "Dark":
    st.markdown("""
        <style>
        .stApp {background-color: #121212; color: white;}
        section[data-testid="stSidebar"] {background-color: #000000 !important;}
        section[data-testid="stSidebar"] * {color: white !important;}
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
            background-color: #2b2b2b !important; color: white !important;
        }
        section[data-testid="stSidebar"] ul[role="listbox"] {
            background-color: #2b2b2b !important; color: white !important;
        }
        label, .stMarkdown, .stText, .stSubheader, .stHeader {
            color: white !important; font-weight: 600;
        }
        .stNumberInput input, .stTextInput input {
            background-color: #e9ecef !important; color: black !important;
        }
        div[data-baseweb="select"] > div {
            background-color: #e9ecef !important; color: black !important;
        }
        ul[role="listbox"] {background-color: white !important; color: black !important;}
        section[data-testid="stFileUploader"] {
            background-color: #1e1e1e !important; border: 1px solid #444 !important;
            border-radius: 10px !important; padding: 10px !important;
        }
        section[data-testid="stFileUploader"] span {color: white !important;}
        section[data-testid="stFileUploader"] button {
            background-color: #28a745 !important; color: black !important;
            font-weight: 600 !important; border-radius: 8px !important;
        }
        </style>
    """, unsafe_allow_html=True)

# =======================================
# GREEN BUTTON STYLE
# =======================================
st.markdown("""
    <style>
    div.stButton > button {
        background-color: #28a745;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        padding: 10px 24px;
        border: none;
    }
    div.stButton > button:hover {
        background-color:#009933";
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# =======================================
# LOAD MODEL
# =======================================
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# =======================================
# FILE LOADER
# =======================================
def load_any_file(uploaded_file):
    try:
        filename = uploaded_file.name.lower()
        if filename.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        elif filename.endswith(".xlsx") or filename.endswith(".xls"):
            return pd.read_excel(uploaded_file)
        elif filename.endswith(".json"):
            return pd.read_json(uploaded_file)
        elif filename.endswith(".txt"):
            return pd.read_csv(uploaded_file, sep=",")
        else:
            st.error("Unsupported file format!")
            return None
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

# =======================================
# NAVIGATION
# =======================================
page = st.sidebar.selectbox("Select Page", ["Prediction", "Analytics", "Bulk Prediction"])

# =======================================
# PREPROCESS
# =======================================
def preprocess_df(df):
    enc = pd.get_dummies(df)
    for c in columns:
        if c not in enc:
            enc[c] = 0
    enc = enc[columns]
    scaled = scaler.transform(enc)
    return scaled

# =======================================
# RISK GAUGE
# =======================================
def risk_gauge(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        title={'text': "Attrition Probability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkred"},
            'steps': [
                {'range': [0, 40], 'color': "#99d98c"},
                {'range': [40, 70], 'color': "#f4d35e"},
                {'range': [70, 100], 'color': "#ee6c4d"}
            ]
        }
    ))
    return fig

# =======================================
# TOP 5 REASONS
# =======================================
def get_reasons(df):
    score = {}
    try:
        if df['OverTime'].iloc[0] == "Yes": score["Overtime work leading to burnout"] = 9
        if df['WorkLifeBalance'].iloc[0] <= 2: score["Poor work-life balance"] = 8
        if df['JobSatisfaction'].iloc[0] <= 2: score["Low job satisfaction"] = 10
        if df['MonthlyIncome'].iloc[0] < 3000: score["Low salary"] = 9
        if df['YearsSinceLastPromotion'].iloc[0] >= 4: score["No promotion for long time"] = 8
        if df['DistanceFromHome'].iloc[0] > 15: score["Long distance from home"] = 7
    except:
        pass

    fallback = ["Career growth issues","Better opportunities elsewhere","Work pressure","Job role mismatch","Lack of recognition"]
    for r in fallback:
        if len(score) >= 5: break
        score[r] = 1

    sorted_reasons = sorted(score.items(), key=lambda x: x[1], reverse=True)
    return [x for x, _ in sorted_reasons[:5]]

# ============================================================
# PAGE 1 — PREDICTION
# ============================================================
if page == "Prediction":

    st.title("Employee Attrition Prediction")

    st.subheader("Personal Details")
    col1, col2 = st.columns(2)

    with col1:
        Age = st.number_input("Age", 18, 60, 30)
        Gender = st.selectbox("Gender", ["Male", "Female"])
        MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

    with col2:
        EducationField = st.selectbox("Education Field",
                                      ["Life Sciences", "Medical", "Marketing",
                                       "Technical Degree", "Human Resources", "Other"])
        DistanceFromHome = st.number_input("Distance From Home (km)", 1, 50, 5)

    st.subheader("Job Details")
    col3, col4 = st.columns(2)

    with col3:
        BusinessTravel = st.selectbox("Business Travel",
                                      ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
        Department = st.selectbox("Department",
                                  ["Sales", "Research & Development", "Human Resources"])
        JobRole = st.selectbox("Job Role",
                               ["Sales Executive", "Research Scientist", "Laboratory Technician",
                                "Manufacturing Director", "Healthcare Representative", "Manager",
                                "Research Director", "Human Resources", "Sales Representative"])
        JobLevel = st.number_input("Job Level", 1, 5, 1)

    with col4:
        JobInvolvement = st.number_input("Job Involvement (1–4)", 1, 4, 3)
        JobSatisfaction = st.number_input("Job Satisfaction (1–4)", 1, 4, 3)
        EnvironmentSatisfaction = st.number_input("Environment Satisfaction (1–4)", 1, 4, 3)
        RelationshipSatisfaction = st.number_input("Relationship Satisfaction (1–4)", 1, 4, 3)
        WorkLifeBalance = st.number_input("Work Life Balance (1–4)", 1, 4, 3)

    st.subheader("Compensation")
    col5, col6 = st.columns(2)

    with col5:
        DailyRate = st.number_input("Daily Rate", 100, 1500, 800)
        HourlyRate = st.number_input("Hourly Rate", 30, 100, 60)
        MonthlyIncome = st.number_input("Monthly Income", 1000, 20000, 5000)

    with col6:
        MonthlyRate = st.number_input("Monthly Rate", 1000, 30000, 20000)
        PercentSalaryHike = st.number_input("Percent Salary Hike", 1, 25, 10)
        OverTime = st.selectbox("Over Time", ["Yes", "No"])

    st.subheader("Experience")
    col7, col8 = st.columns(2)

    with col7:
        TotalWorkingYears = st.number_input("Total Working Years", 0, 40, 5)
        YearsAtCompany = st.number_input("Years at Company", 0, 40, 3)
        YearsInCurrentRole = st.number_input("Years in Current Role", 0, 20, 2)

    with col8:
        YearsSinceLastPromotion = st.number_input("Years Since Last Promotion", 0, 20, 1)
        YearsWithCurrManager = st.number_input("Years With Current Manager", 0, 20, 2)
        NumCompaniesWorked = st.number_input("Num Companies Worked", 0, 10, 1)
        TrainingTimesLastYear = st.number_input("Training Times Last Year", 0, 10, 2)

    input_df = pd.DataFrame({
        "Age": [Age], "Gender": [Gender], "MaritalStatus": [MaritalStatus],
        "EducationField": [EducationField], "DistanceFromHome": [DistanceFromHome],
        "BusinessTravel": [BusinessTravel], "Department": [Department], "JobRole": [JobRole],
        "JobLevel": [JobLevel], "JobInvolvement": [JobInvolvement], "JobSatisfaction": [JobSatisfaction],
        "EnvironmentSatisfaction": [EnvironmentSatisfaction], "RelationshipSatisfaction": [RelationshipSatisfaction],
        "WorkLifeBalance": [WorkLifeBalance], "DailyRate": [DailyRate], "HourlyRate": [HourlyRate],
        "MonthlyIncome": [MonthlyIncome], "MonthlyRate": [MonthlyRate], "PercentSalaryHike": [PercentSalaryHike],
        "OverTime": [OverTime], "TotalWorkingYears": [TotalWorkingYears], "YearsAtCompany": [YearsAtCompany],
        "YearsInCurrentRole": [YearsInCurrentRole], "YearsSinceLastPromotion": [YearsSinceLastPromotion],
        "YearsWithCurrManager": [YearsWithCurrManager], "NumCompaniesWorked": [NumCompaniesWorked],
        "TrainingTimesLastYear": [TrainingTimesLastYear],
    })

    colA, colB = st.columns(2)

    with colA:
        if st.button("Predict Attrition"):
            with st.spinner("Analyzing employee data..."):
                scaled = preprocess_df(input_df)
                prob = model.predict_proba(scaled)[0][1] * 100

                st.success(f"Attrition Probability: {prob:.2f}%")
                st.plotly_chart(risk_gauge(prob), use_container_width=True)

                if prob > 35:
                    st.subheader("Top 5 Reasons for Attrition")
                    reasons = get_reasons(input_df)
                    for r in reasons:
                        st.write(f"- {r}")

    with colB:
        if st.button("Reset Form"):
            st.rerun()

# ============================================================
# PAGE 2 — ANALYTICS
# ============================================================
elif page == "Analytics":
    st.title("Employee Analytics Dashboard")

    uploaded = st.file_uploader("Upload dataset", type=["csv", "xlsx", "json"])

    if uploaded:
        df = load_any_file(uploaded)
        st.success("Dataset loaded successfully!")

        if "Attrition" in df.columns:
            st.subheader("Attrition Distribution")
            st.plotly_chart(px.pie(df, names="Attrition"), use_container_width=True)

        if "Gender" in df.columns:
            st.subheader("Employees by Gender")
            st.plotly_chart(px.pie(df, names="Gender"), use_container_width=True)

        if "Department" in df.columns:
            st.subheader("Employees by Department")
            st.plotly_chart(px.pie(df, names="Department"), use_container_width=True)

        if "EducationField" in df.columns:
            st.subheader("Employees by Education Field")
            st.plotly_chart(px.pie(df, names="EducationField"), use_container_width=True)

        if "JobRole" in df.columns:
            st.subheader("Employees by Job Role")
            st.plotly_chart(px.pie(df, names="JobRole"), use_container_width=True)

        st.subheader("Dataset Sample")
        rows = st.slider("Number of rows to display", 5, 100, 20)
        st.dataframe(df.head(rows), use_container_width=True)

# ============================================================
# PAGE 3 — BULK PREDICTION
# ============================================================
elif page == "Bulk Prediction":
    st.title("Bulk Prediction")

    uploaded = st.file_uploader("Upload employee dataset:", type=["csv", "xlsx"])

    if uploaded:
        df_bulk = load_any_file(uploaded)

        if df_bulk is not None:
            scaled = preprocess_df(df_bulk)
            preds = model.predict(scaled)
            probs = model.predict_proba(scaled)[:, 1] * 100

            risk_levels = []
            for p in probs:
                if p <= 35:
                    risk_levels.append("Low Risk")
                elif p <= 70:
                    risk_levels.append("Medium Risk")
                else:
                    risk_levels.append("High Risk")

            out = df_bulk.copy()
            out["Prediction"] = preds
            out["Probability"] = np.round(probs, 2)
            out["Risk Level"] = risk_levels

            # SORT HIGH TO LOW RISK
            out = out.sort_values(by="Probability", ascending=False)

            st.subheader("All Employees Sorted by Attrition Risk")
            st.dataframe(out, use_container_width=True)

            st.subheader("High Risk Employees")
            st.dataframe(out[out["Risk Level"] == "High Risk"], use_container_width=True)

            st.download_button(
                "Download Results (Sorted by Risk)",
                out.to_csv(index=False).encode("utf-8"),
                "bulk_predictions_sorted.csv"
            )