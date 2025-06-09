import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Setup halaman
st.set_page_config(page_title="Klasifikasi Padi", layout="wide")
st.title("Aplikasi Klasifikasi Padi dengan Regresi Logistik")

# Fungsi load data
@st.cache_data
def load_data():
    df = pd.read_excel("hasil klasifikasi final.xlsx")
    df.columns = [col.strip().lower().replace(" ", "_").replace("%", "persen") for col in df.columns]
    df.dropna(inplace=True)
    return df

df = load_data()

st.subheader("🔍 Data Awal")
st.dataframe(df.head(), use_container_width=True)

# Visualisasi distribusi kelas
st.subheader("Distribusi Kelas")
sns.countplot(x="klasifikasi", data=df, palette="Set2")
st.pyplot()

# Split fitur dan label
X = df.drop("klasifikasi", axis=1)
y = df["klasifikasi"]

# Normalisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

# Evaluasi model
st.subheader("Evaluasi Model")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.write(f"**Akurasi:** {accuracy_score(y_test, y_pred):.2f}")

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
st.pyplot()

# Form input prediksi manual
st.subheader("Prediksi Klasifikasi Baru")
input_data = {}

for col in X.columns:
    input_data[col] = st.number_input(f"{col}", step=0.1)

if st.button("Prediksi"):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)
    st.success(f"Prediksi klasifikasi: **{pred[0]}**")
