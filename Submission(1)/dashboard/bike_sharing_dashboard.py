import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime

# Judul Dashboard
st.title("📊 Bike Sharing Data Analysis")

# Load Dataset
@st.cache_data
def load_data():
    day_df = pd.read_csv("C:\\Users\\Leviathans\\Downloads\\Bike-sharing-dataset\\dashboard\\day.csv")
    hour_df = pd.read_csv("C:\\Users\\Leviathans\\Downloads\\Bike-sharing-dataset\\dashboard\\hour.csv")
    
    day_df["dteday"] = pd.to_datetime(day_df["dteday"])
    hour_df["dteday"] = pd.to_datetime(hour_df["dteday"])
    return day_df, hour_df

# Load data
day_df, hour_df = load_data()

# Sidebar - Filter Data
st.sidebar.header("📆 Filter Data")
start_date = st.sidebar.date_input("Mulai Tanggal", day_df["dteday"].min().date())
end_date = st.sidebar.date_input("Sampai Tanggal", day_df["dteday"].max().date())

# Filter berdasarkan rentang tanggal
mask = (day_df["dteday"].dt.date >= start_date) & (day_df["dteday"].dt.date <= end_date)
day_df = day_df[mask]

# Filter untuk hour_df
hour_df = hour_df[(hour_df["dteday"].dt.date >= start_date) & (hour_df["dteday"].dt.date <= end_date)]

selected_season = st.sidebar.selectbox("Pilih Musim", ["Semua", "Spring", "Summer", "Fall", "Winter"])
season_mapping = {"Spring": 1, "Summer": 2, "Fall": 3, "Winter": 4}

if selected_season != "Semua":
    day_df = day_df[day_df["season"] == season_mapping[selected_season]]
    hour_df = hour_df[hour_df["season"] == season_mapping[selected_season]]

#ren Peminjaman Sepeda (Harian, Bulanan, Musiman)
st.subheader("📈 Tren Peminjaman Sepeda")

#Bulanan
day_df["month"] = day_df["dteday"].dt.month
monthly_counts = day_df.groupby("month")["cnt"].sum()

fig_monthly = px.bar(
    x=monthly_counts.index, y=monthly_counts.values,
    labels={'x': 'Bulan', 'y': 'Total Peminjaman'},
    title="Jumlah Peminjaman Sepeda per Bulan",
    text_auto=True,
    color=monthly_counts.values,
    color_continuous_scale="Blues"
)
st.plotly_chart(fig_monthly)

#Musiman
season_labels = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
day_df["season_label"] = day_df["season"].map(season_labels)
fig_season = px.box(day_df, x="season_label", y="cnt", color="season_label", title="Distribusi Peminjaman Sepeda Berdasarkan Musim")
st.plotly_chart(fig_season)

#Analisis Peminjaman Berdasarkan Jam
st.subheader("⏰ Analisis Peminjaman Sepeda Berdasarkan Jam")
hourly_counts = hour_df.groupby("hr")["cnt"].mean()
fig_hourly = px.line(
    x=hourly_counts.index, y=hourly_counts.values,
    labels={'x': 'Jam', 'y': 'Rata-rata Peminjaman'},
    title="Rata-rata Peminjaman Sepeda per Jam",
    markers=True
)
st.plotly_chart(fig_hourly)

#Hubungan Cuaca dengan Peminjaman Sepeda
st.subheader("🌤️ Pengaruh Cuaca terhadap Peminjaman Sepeda")
weather_corr = day_df[["temp", "hum", "windspeed", "cnt"]].corr()
fig_corr = px.imshow(weather_corr, text_auto=True, title="Korelasi Cuaca vs. Peminjaman Sepeda")
st.plotly_chart(fig_corr)

#RFM Analysis
st.subheader("📊 Analisis RFM (Recency, Frequency, Monetary)")
max_date = day_df["dteday"].max()
day_df["Recency"] = (max_date - day_df["dteday"]).dt.days
day_df["Frequency"] = day_df["casual"] + day_df["registered"]
day_df["Monetary"] = day_df["cnt"]

day_df["R_Score"] = pd.qcut(day_df["Recency"], 5, labels=[5, 4, 3, 2, 1])
day_df["F_Score"] = pd.qcut(day_df["Frequency"], 5, labels=[1, 2, 3, 4, 5])
day_df["M_Score"] = pd.qcut(day_df["Monetary"], 5, labels=[1, 2, 3, 4, 5])
day_df["RFM_Score"] = day_df["R_Score"].astype(int) + day_df["F_Score"].astype(int) + day_df["M_Score"].astype(int)

st.dataframe(day_df[["dteday", "Recency", "Frequency", "Monetary", "RFM_Score"]].head())
fig_rfm = px.histogram(day_df, x="RFM_Score", nbins=10, title="Distribusi RFM Score")
st.plotly_chart(fig_rfm)

#Kesimpulan
st.subheader("📌 Kesimpulan")
st.markdown("""
- **Peminjaman sepeda meningkat pada musim panas (Summer) dengan rata-rata 4.504 pengguna per hari.**
- **Suhu memiliki korelasi positif (r = 0.63) dengan jumlah peminjaman, sedangkan kelembaban dan kecepatan angin berdampak kecil.**
- **Sebagian besar pengguna memiliki skor RFM rendah, menunjukkan bahwa banyak pengguna yang tidak aktif.**
""")

#Footer
st.markdown("---")
st.markdown("📊 Dashboard dibuat oleh **[MUHAMMAD AGUSRIANSYAH]**")
