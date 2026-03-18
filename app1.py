import streamlit as st
import requests
import joblib
import pandas as pd
import numpy as np
import plotly.express as px

# =========================
# 🎨 FINAL UI (SUN + HEATWAVE)
# =========================
st.markdown("""
<style>

/* 🌅 Heatwave Background */
.stApp {
    background: linear-gradient(135deg, #ff7e5f, #ff512f, #1a1a2e);
    color: white;
}

/* ☀️ Sun Animation */
.sun {
    position: fixed;
    top: 60px;
    right: 80px;
    width: 120px;
    height: 120px;
    background: radial-gradient(circle, #ffcc00, #ff6600);
    border-radius: 50%;
    box-shadow: 0 0 80px rgba(255,165,0,0.8);
    animation: pulse 3s infinite ease-in-out;
    z-index: -1;
}

@keyframes pulse {
    0% { transform: scale(1); opacity: 0.8; }
    50% { transform: scale(1.2); opacity: 1; }
    100% { transform: scale(1); opacity: 0.8; }
}

/* Glass UI */
div[data-testid="stMetric"], .stAlert, .stSubheader {
    background: rgba(0, 0, 0, 0.5);
    padding: 15px;
    border-radius: 12px;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #ff512f, #ff9966);
    color: white;
    border-radius: 10px;
}

/* Input */
.stTextInput input {
    background-color: rgba(0,0,0,0.5);
    color: white;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: white !important;
    font-weight: 600;
}
button[data-baseweb="tab"][aria-selected="true"] {
    border-bottom: 3px solid #ffcc00;
    color: #ffcc00 !important;
}

</style>

<div class="sun"></div>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.markdown("""
<h1 style='white-space: nowrap;'>🌞 HeatWave Insight System</h1>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
model = joblib.load("xgboost_model.pkl")
features = joblib.load("model_features.pkl")

API_KEY = "92c2e0509859c54d808577aac9ae09ea"

# =========================
# WEATHER FUNCTION
# =========================
def get_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    data = requests.get(url).json()

    if "main" not in data:
        return None

    return {
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "rainfall": data.get("rain", {}).get("1h", 0),
        "wind_speed": data["wind"]["speed"] * 3.6,
        "pressure": data["main"]["pressure"],
        "lat": data["coord"]["lat"],
        "lon": data["coord"]["lon"]
    }

# =========================
# PREDICTION FUNCTION
# =========================
def predict_heatwave(weather):

    df = pd.DataFrame([{
        "latitude": weather["lat"],
        "longitude": weather["lon"],
        "max_temperature": weather["temperature"],
        "min_temperature": weather["temperature"],
        "max_humidity": weather["humidity"],
        "min_humidity": weather["humidity"] - 10,
        "wind_speed": weather["wind_speed"],
        "pressure_surface_level": weather["pressure"],
        "cloud_cover": 20,
        "visibility": 10,
        "uv_index": 7,
        "solar_radiation": 500
    }])

    df = df[features]

    prob = model.predict_proba(df)[0][1] * 100

    # 🔥 Improve realism
    prob += (weather["temperature"] - 25) * 2
    prob = max(0, min(prob, 100))

    if prob > 70:
        level = "Severe"
    elif prob > 50:
        level = "High"
    elif prob > 30:
        level = "Moderate"
    else:
        level = "Low"

    pred = "🔥 Heatwave" if prob > 50 else "✅ No Heatwave"

    return pred, prob, level

# =========================
# TABS
# =========================
tabs = st.tabs(["🔍 Prediction", "🔆 Heatmap"])

# =========================
# TAB 1: PREDICTION
# =========================
with tabs[0]:

    city = st.text_input("Enter City Name")

    if st.button("Predict"):

        weather = get_weather(city)

        if weather is None:
            st.error("❌ City not found")
        else:
            pred, prob, level = predict_heatwave(weather)

            st.success("Prediction Complete!")

            st.subheader("Result")
            st.write(f"📍 City: {city}")
            st.write(f"🌡️ Prediction: {pred}")
            st.write(f"📊 Probability: {round(prob,2)}%")
            st.write(f"⚠️ Risk Level: {level}")

            # Live weather
            st.subheader("🌤️ Live Weather")
            c1, c2, c3 = st.columns(3)
            c1.metric("Temp", f"{weather['temperature']}°C")
            c2.metric("Humidity", f"{weather['humidity']}%")
            c3.metric("Wind", f"{weather['wind_speed']:.2f}")

            # 24 hour graph
            st.subheader("🌡️ 24-Hour Projection")

            hours = list(range(24))
            temps = [
                weather["temperature"] - 3 + 5 * np.sin((h - 6) / 24 * 2 * np.pi)
                for h in hours
            ]

            df_temp = pd.DataFrame({"Hour": hours, "Temp": temps})
            st.plotly_chart(px.area(df_temp, x="Hour", y="Temp"))

            # Explanation
            st.subheader("📊 Why this Prediction")

            explain = {
                "Temperature": weather["temperature"] * 0.4,
                "Humidity": (100 - weather["humidity"]) * 0.3,
                "Wind": (30 - weather["wind_speed"]) * 0.2,
                "Pressure": abs(1010 - weather["pressure"]) * 0.1
            }

            df_explain = pd.DataFrame({
                "Feature": list(explain.keys()),
                "Impact": list(explain.values())
            })

            st.plotly_chart(px.bar(df_explain, x="Feature", y="Impact"))

# =========================
# TAB 2: HEATMAP
# =========================
with tabs[1]:

    cities = ["Delhi","Mumbai","Chennai","Ahmedabad","Bangalore","Kolkata","Hyderabad"]

    results = []

    for city in cities:
        weather = get_weather(city)
        if weather:
            _, prob, _ = predict_heatwave(weather)
            results.append({
                "City": city,
                "lat": weather["lat"],
                "lon": weather["lon"],
                "risk": prob
            })

    df_map = pd.DataFrame(results)

    fig = px.density_mapbox(
        df_map,
        lat="lat",
        lon="lon",
        z="risk",
        radius=30,
        center=dict(lat=22.5, lon=78.9),
        zoom=4,
        mapbox_style="open-street-map",
        color_continuous_scale="YlOrRd"
    )

    st.plotly_chart(fig)
