import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Collision Estimator", layout="centered")
st.title("Εκτιμητής Κρούσης")

@st.cache_data
def load_and_prepare_data():
    # Αντικατάσταση με το όνομα του αρχείου ή χρησιμοποιούμε δεδομένα inline
    data = {
        "Mass": [0.417]*20,
        "Width": [0.0127]*20,
        "Distance": [0.60,0.50,0.40,0.30,0.25,0.20,0.15,0.60,0.60,0.60,0.60,0.60,0.60,0.60,0.60,0.60,0.60,0.60,0.60,0.60],
        "Material": ["0","0","0","0","0","0","0","Μαλακό Αφρολέξ","Σκληρό Αφρολέξ","Φελιζόλ","Φελλός","Συνθετικό","Τσόχα","Τσόχα","Τσόχα","Τσόχα","Τσόχα","Τσόχα","Τσόχα","Τσόχα"],
        "Sheets": [0,0,0,0,0,0,0,1,1,1,1,1,0,2,4,6,8,10,12,14],
        "Thickness": [0,0,0,0,0,0,0,0.09,0.09,0.09,0.09,0.09,0,0,0,0,0,0,0,0],
        "Force_Mean": [46.2,43.2,37.9,32.7,29.8,27.0,23.8,38.3,16.2,17.2,40.9,16.7,46.4,41.9,34.2,31.1,25.8,19.7,15.9,13.0],
        "Time_Mean": [0.012,0.013,0.013,0.012,0.013,0.013,0.013,0.038,0.042,0.040,0.014,0.045,0.013,0.022,0.032,0.038,0.048,0.061,0.070,0.089],
        "Velocity_Initial": [0.41,0.37,0.33,0.28,0.25,0.23,0.20,0.41,0.42,0.41,0.42,0.42,0.43,0.43,0.42,0.41,0.42,0.42,0.32,0.44]
    }
    df = pd.DataFrame(data)
    return df

df = load_and_prepare_data()

encoder = LabelEncoder()
df["Material_encoded"] = encoder.fit_transform(df["Material"])

X = df[["Material_encoded", "Distance", "Thickness", "Mass", "Velocity_Initial"]]
y_force = df["Force_Mean"]
y_time = df["Time_Mean"]

@st.cache_resource
def train_models():
    X_train, X_test, yF_train, yF_test = train_test_split(X, y_force, test_size=0.2, random_state=42)
    _, _, yT_train, yT_test = train_test_split(X, y_time, test_size=0.2, random_state=42)
    model_force = RandomForestRegressor(n_estimators=300, random_state=42)
    model_time = RandomForestRegressor(n_estimators=300, random_state=42)
    model_force.fit(X_train, yF_train)
    model_time.fit(X_train, yT_train)
    return model_force, model_time

model_force, model_time = train_models()

st.header("Είσοδοι")
material = st.selectbox("Υλικό Προφυλακτήρα:", encoder.classes_.tolist())
distance = st.slider("Απόσταση (m):", 0.1, 1.0, 0.6, 0.01)
thickness = st.slider("Πάχος Προφυλακτήρα (m):", 0.0, 0.1, 0.0, 0.001)
mass = st.slider("Μάζα (kg):", 0.1, 1.0, 0.417, 0.01)
velocity = st.slider("Αρχική Ταχύτητα (m/s):", 0.1, 1.0, 0.4, 0.01)

if st.button("Υπολογισμός"):
    material_encoded = encoder.transform([material])[0]
    input_df = pd.DataFrame([{
        "Material_encoded": material_encoded,
        "Distance": distance,
        "Thickness": thickness,
        "Mass": mass,
        "Velocity_Initial": velocity
    }])
    pred_force = float(model_force.predict(input_df)[0])
    pred_time = float(model_time.predict(input_df)[0])
    delta_p = pred_force * pred_time
    delta_v = delta_p / mass if mass != 0 else np.nan

    st.subheader("Αποτελέσματα")
    st.metric("Μέγιστη Εκτιμώμενη Δύναμη", f"{pred_force:.2f} N")
    st.metric("Εκτιμώμενη Διαφορά Ταχύτητας (Δv)", f"{delta_v:.3f} m/s")
    st.metric("Εκτιμώμενη Διαφορά Ορμής (Δp)", f"{delta_p:.3f} N·s")

st.header("Heatmap Συσχετίσεων")
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax)
ax.set_title("Correlation Heatmap Μεταβλητών")
st.pyplot(fig)

with st.expander("Προβολή Δεδομένων"):
    st.dataframe(df)
