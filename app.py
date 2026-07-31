import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os
from datetime import datetime

# ======================================================
# PAGE CONFIGURATION
# ======================================================

st.set_page_config(
    page_title="Smart City Traffic Forecasting",
    page_icon="🚦",
    layout="wide"
)

# ======================================================
# SIDEBAR THEME
# ======================================================

theme = st.sidebar.toggle("🌙 Dark Mode", value=True)

if theme:
    bg_color = "#0E1117"
    text_color = "#FFFFFF"
    card_color = "#1E1E1E"
else:
    bg_color = "#FFFFFF"
    text_color = "#000000"
    card_color = "#F3F4F6"

st.markdown(
    f"""
<style>

.stApp {{
    background-color:{bg_color};
    color:{text_color};
}}

section[data-testid="stSidebar"] {{
    background-color:{card_color};
}}

h1,h2,h3,h4,h5,h6,p,label {{
    color:{text_color} !important;
}}

div[data-testid="stMetric"] {{
    background:{card_color};
    border-radius:15px;
    padding:18px;
    border:1px solid #2563eb;
}}

.stButton>button {{
    width:100%;
    border-radius:10px;
    background:#2563eb;
    color:white;
    font-weight:bold;
}}

</style>
""",
    unsafe_allow_html=True,
)

# ======================================================
# FILE PATHS
# ======================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "train_aWnotuB.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model.pkl"
)

# ======================================================
# LOAD DATA
# ======================================================

if not os.path.exists(TRAIN_PATH):
    st.error("Training dataset not found.")
    st.stop()

if not os.path.exists(MODEL_PATH):
    st.error("model.pkl not found.")
    st.stop()

train = pd.read_csv(TRAIN_PATH)
model = joblib.load(MODEL_PATH)

# ======================================================
# FEATURE ENGINEERING
# ======================================================

train["DateTime"] = pd.to_datetime(train["DateTime"])

train["Year"] = train["DateTime"].dt.year
train["Month"] = train["DateTime"].dt.month
train["Day"] = train["DateTime"].dt.day
train["Hour"] = train["DateTime"].dt.hour
train["DayOfWeek"] = train["DateTime"].dt.dayofweek
train["Weekend"] = (train["DayOfWeek"] >= 5).astype(int)
train["Quarter"] = train["DateTime"].dt.quarter
train["WeekOfYear"] = train["DateTime"].dt.isocalendar().week.astype(int)

# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.title("🚦 Smart City")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Traffic Prediction",
        "Batch Prediction",
    ],
)

# ======================================================
# DASHBOARD
# ======================================================

def dashboard():

    st.markdown(
        """
<div style="
padding:25px;
border-radius:20px;
background:linear-gradient(90deg,#2563eb,#7c3aed);
text-align:center;
color:white;
">

<h1>🚦 Smart City Traffic Forecasting</h1>
<h4>AI Powered Traffic Analytics Dashboard</h4>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "📊 Total Records",
            f"{len(train):,}"
        )

    with col2:
        st.metric(
            "🚗 Average Vehicles",
            round(train["Vehicles"].mean(), 2)
        )

    with col3:
        st.metric(
            "🚦 Maximum Vehicles",
            train["Vehicles"].max()
        )

    with col4:
        st.metric(
            "🤖 Model",
            "Random Forest"
        )

    st.divider()

    st.subheader("Average Hourly Traffic")

    hourly = (
        train.groupby("Hour")["Vehicles"]
        .mean()
        .reset_index()
    )

    fig = px.line(
        hourly,
        x="Hour",
        y="Vehicles",
        markers=True,
        title="Average Traffic by Hour",
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader("Traffic by Junction")

    junction = (
        train.groupby("Junction")["Vehicles"]
        .mean()
        .reset_index()
    )

    fig2 = px.bar(
        junction,
        x="Junction",
        y="Vehicles",
        color="Vehicles",
        title="Average Vehicles at Each Junction",
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.divider()

    st.subheader("Monthly Traffic Trend")

    monthly = (
        train.groupby("Month")["Vehicles"]
        .mean()
        .reset_index()
    )

    fig3 = px.area(
        monthly,
        x="Month",
        y="Vehicles",
        title="Monthly Average Traffic",
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )
# ======================================================
# TRAFFIC PREDICTION
# ======================================================

def traffic_prediction():

    st.title("🚗 Live Traffic Prediction")

    col1, col2 = st.columns(2)

    with col1:

        junction = st.selectbox(
            "Select Junction",
            [1, 2, 3, 4]
        )

        year = st.number_input(
            "Year",
            min_value=2015,
            max_value=2035,
            value=2017
        )

        month = st.slider(
            "Month",
            1,
            12,
            7
        )

        day = st.slider(
            "Day",
            1,
            31,
            15
        )

    with col2:

        hour = st.slider(
            "Hour",
            0,
            23,
            18
        )

        dayofweek = st.selectbox(
            "Day of Week",
            [
                0, 1, 2, 3, 4, 5, 6
            ],
            format_func=lambda x: [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday"
            ][x]
        )

    weekend = 1 if dayofweek >= 5 else 0

    quarter = ((month - 1) // 3) + 1

    try:
        weekofyear = datetime(
            year,
            month,
            day
        ).isocalendar()[1]

    except ValueError:

        st.error("Invalid Date Selected")
        return

    st.write("")

    if st.button("🚦 Predict Traffic"):

        input_df = pd.DataFrame({

            "Junction": [junction],
            "Year": [year],
            "Month": [month],
            "Day": [day],
            "Hour": [hour],
            "DayOfWeek": [dayofweek],
            "Weekend": [weekend],
            "Quarter": [quarter],
            "WeekOfYear": [weekofyear]

        })

        prediction = model.predict(input_df)[0]

        st.success("Prediction Completed Successfully!")

        st.metric(
            label="🚗 Predicted Vehicles",
            value=f"{round(prediction)} Vehicles"
        )

        st.progress(
            min(
                int(prediction),
                100
            )
        )

        st.write("")

        if prediction < 20:

            st.success(
                "🟢 Traffic Status : LOW"
            )

        elif prediction < 40:

            st.warning(
                "🟡 Traffic Status : MODERATE"
            )

        else:

            st.error(
                "🔴 Traffic Status : HEAVY"
            )

        st.divider()

        st.subheader("Prediction Summary")

        summary = pd.DataFrame({

            "Feature": [
                "Junction",
                "Year",
                "Month",
                "Day",
                "Hour",
                "Weekend"
            ],

            "Value": [
                junction,
                year,
                month,
                day,
                hour,
                weekend
            ]

        })

        st.dataframe(
            summary,
            use_container_width=True
        )
# ======================================================
# BATCH PREDICTION
# ======================================================

def batch_prediction():

    st.title("📂 Batch Traffic Prediction")

    st.write(
        "Upload a CSV file containing **DateTime** and **Junction** columns."
    )

    uploaded_file = st.file_uploader(
        "Choose CSV File",
        type=["csv"]
    )

    if uploaded_file is None:
        return

    try:
        df = pd.read_csv(uploaded_file)

    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return

    required_columns = ["DateTime", "Junction"]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        st.error(
            f"Missing required column(s): {', '.join(missing)}"
        )
        return

    # -----------------------------
    # Feature Engineering
    # -----------------------------

    df["DateTime"] = pd.to_datetime(df["DateTime"])

    df["Year"] = df["DateTime"].dt.year
    df["Month"] = df["DateTime"].dt.month
    df["Day"] = df["DateTime"].dt.day
    df["Hour"] = df["DateTime"].dt.hour
    df["DayOfWeek"] = df["DateTime"].dt.dayofweek
    df["Weekend"] = (df["DayOfWeek"] >= 5).astype(int)
    df["Quarter"] = df["DateTime"].dt.quarter
    df["WeekOfYear"] = (
        df["DateTime"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    features = [
        "Junction",
        "Year",
        "Month",
        "Day",
        "Hour",
        "DayOfWeek",
        "Weekend",
        "Quarter",
        "WeekOfYear",
    ]

    predictions = model.predict(df[features])

    df["Predicted Vehicles"] = predictions.round().astype(int)

    st.success("Prediction Completed!")

    st.subheader("Prediction Results")

    st.dataframe(
        df,
        use_container_width=True
    )

    st.divider()

    st.subheader("Traffic Distribution")

    fig = px.histogram(
        df,
        x="Predicted Vehicles",
        nbins=20,
        title="Predicted Traffic Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader("Predicted Traffic by Junction")

    junction_avg = (
        df.groupby("Junction")["Predicted Vehicles"]
        .mean()
        .reset_index()
    )

    fig2 = px.bar(
        junction_avg,
        x="Junction",
        y="Predicted Vehicles",
        color="Predicted Vehicles",
        title="Average Predicted Vehicles"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download Prediction CSV",
        data=csv,
        file_name="Predicted_Traffic.csv",
        mime="text/csv"
    )
    # ======================================================
# FOOTER
# ======================================================

def footer():

    st.divider()

    st.markdown(
        """
### 👨‍💻 Smart City Traffic Forecasting

**Machine Learning Model:** Random Forest Regressor

### 🛠️ Technologies Used

- Python
- Pandas
- Scikit-Learn
- Plotly
- Streamlit
"""
    )

    st.markdown(
        """
<div style="text-align:center;
padding:20px;
font-size:16px;">

🚦 <b>Smart City Traffic Forecasting</b><br><br>

Developed using
<b>Python | Pandas | Scikit-Learn | Plotly | Streamlit</b>

<br><br>

© 2026 Internship Project

</div>
""",
        unsafe_allow_html=True
    )

# ======================================================
# MAIN APPLICATION
# ======================================================

if page == "Dashboard":

    dashboard()

elif page == "Traffic Prediction":

    traffic_prediction()

elif page == "Batch Prediction":

    batch_prediction()

footer()