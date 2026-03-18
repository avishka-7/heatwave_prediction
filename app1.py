import streamlit as st
import requests
import joblib
import pandas as pd
import numpy as np
import plotly.express as px

# =========================
st.markdown("""
<style>

/* 🔥 Heatwave Theme Background */
.stApp {
    background: radial-gradient(circle at top, #ff512f, #dd2476, #1a1a2e);
    color: white;
}

/* ☀️ Heat glow effect */
.stApp::before {
    content: "";
    position: fixed;
    top: -100px;
    left: 50%;
    transform: translateX(-50%);
    width: 600px;
    height: 600px;
    background: radial-gradient(circle, rgba(255,165,0,0.4), transparent 70%);
    filter: blur(120px);
    z-index: -1;
}

/* ✨ Glass Cards */
div[data-testid="stMetric"], .stAlert, .stSubheader {
    background: rgba(0, 0, 0, 0.5);
    padding: 15px;
    border-radius: 12px;
    backdrop-filter: blur(10px);
}

/* 🔥 Button */
.stButton>button {
    background: linear-gradient(90deg, #ff512f, #ff9966);
    color: white;
    border-radius: 10px;
    border: none;
}

/* 🔤 Input */
.stTextInput input {
    background-color: rgba(0,0,0,0.5);
    color: white;
    border-radius: 8px;
}

/* 🧾 Headers */
h1, h2, h3 {
    color: #ffffff;
}

</style>
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
    response = requests.get(url)
    data = response.json()

    if response.status_code != 200:
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

    input_data = {
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
    }

    df = pd.DataFrame([input_data])
    df = df[features]

    prob = model.predict_proba(df)[0][1] * 100

    if prob > 70:
        level = "Severe"
    elif prob > 50:
        level = "High"
    elif prob > 30:
        level = "Moderate"
    else:
        level = "Low"

    prediction = "🔥 Heatwave" if prob > 50 else "✅ No Heatwave"

    return prediction, prob, level

# =========================
# UI
# =========================
st.title("🌍 Climate Heatwave Prediction System")

tabs = st.tabs([
    "🔍 Prediction",
    "🗺️ Geospatial Map",
    "🌆 Multi-City Monitor"
])

# =========================
# TAB 1: PREDICTION
# =========================
with tabs[0]:

    st.header("Heatwave Prediction")

    city = st.text_input("Enter City Name").strip().title()

    if st.button("Predict"):

        weather = get_weather(city)

        if weather is None:
            st.error("❌ City not found")
        else:
            pred, prob, level = predict_heatwave(weather)

            st.success("Prediction Complete!")

            # RESULT (SAME AS YOUR STYLE)
            st.subheader("Result")
            st.write(f"📍 City: {city}")
            st.write(f"🌡️ Prediction: {pred}")
            st.write(f"📊 Heatwave Probability: {round(prob,2)} %")
            st.write(f"⚠️ Risk Level: {level}")
            st.write(f"🌍 Coordinates: ({weather['lat']}, {weather['lon']})")

            if prob > 60:
                st.error("⚠️ High Heatwave Risk! Stay Alert")
            elif prob > 40:
                st.warning("Moderate Risk Conditions")
            else:
                st.success("Conditions are safe")

            # LIVE WEATHER
            st.subheader("🌤️ Live Weather Report")

            c1, c2, c3 = st.columns(3)
            c1.metric("Temperature", f"{weather['temperature']} °C")
            c2.metric("Humidity", f"{weather['humidity']} %")
            c3.metric("Wind Speed", f"{weather['wind_speed']:.2f} km/h")

            # 24-HOUR GRAPH
            st.subheader("🌡️ 24-Hour Temperature Projection")

            hours = list(range(24))
            base_temp = weather["temperature"]

            temps = [
                base_temp - 3 + 5 * np.sin((h - 6) / 24 * 2 * np.pi)
                for h in hours
            ]

            temp_df = pd.DataFrame({
                "Hour": hours,
                "Temperature": temps
            })

            fig = px.area(temp_df, x="Hour", y="Temperature")
            st.plotly_chart(fig)

# =========================
# TAB 2: MAP
# =========================
with tabs[1]:

    st.header("Geospatial Heatwave Risk Map")

    city_map = st.text_input("Enter City for Map").strip().title()

    if st.button("Show Map"):

        weather = get_weather(city_map)

        if weather:
            pred, prob, level = predict_heatwave(weather)

            map_df = pd.DataFrame({
                "lat": [weather["lat"]],
                "lon": [weather["lon"]],
                "risk": [prob]
            })

            fig = px.scatter_mapbox(
                map_df,
                lat="lat",
                lon="lon",
                size="risk",
                color="risk",
                zoom=4,
                mapbox_style="open-street-map"
            )

            st.plotly_chart(fig)

# =========================
# TAB 3: MULTI CITY
# =========================
with tabs[2]:

    st.header("Multi-City Heatwave Monitor")

    cities = ["Delhi","Mumbai","Chennai","Ahmedabad","Bangalore","Kolkata","Hyderabad"]

    if st.button("Run Monitoring"):

        results = []

        for city in cities:
            weather = get_weather(city)

            if weather:
                pred, prob, level = predict_heatwave(weather)

                results.append({
                    "City": city,
                    "Temp": weather["temperature"],
                    "Risk (%)": round(prob, 2),
                    "Level": level
                })

        df = pd.DataFrame(results).sort_values(by="Risk (%)", ascending=False)

        st.dataframe(df)

        fig = px.bar(df, x="City", y="Risk (%)", color="Level")
        st.plotly_chart(fig)
