# dashboard_forecasting_hw.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(
    layout="wide",
    page_title="DSS - Holt-Winters Forecasting & Inventory Optimization"
)

st.title("Decision Support System - Holt–Winters Forecasting & Inventory Optimization")

st.markdown("""
Dashboard ini mendukung :
- Forecasting permintaan menggunakan **Holt–Winters**
- Perhitungan **Safety Stock, Reorder Point, dan EOQ**
- Analisis pendukung keputusan persediaan bahan baku
""")

# =========================
# Upload File
# =========================
uploaded_file = st.file_uploader(
    "Upload File Excel Transaksi Bahan Baku",
    type=["xlsx", "xls"]
)

if uploaded_file is None:
    st.info("Silakan upload file Excel terlebih dahulu.")
    st.stop()

# =========================
# Load Data
# =========================
@st.cache_data
def load_data(file):
    stok = pd.read_excel(file, sheet_name="StokGudang")
    pemb = pd.read_excel(file, sheet_name="Pembelian")
    pakai = pd.read_excel(file, sheet_name="Penggunaan")
    return stok, pemb, pakai

stok_df, pemb_df, pakai_df = load_data(uploaded_file)

# =========================
# Preprocessing
# =========================
for df in [stok_df, pemb_df, pakai_df]:
    if {"Tahun", "Bulan"}.issubset(df.columns):
        df["Tanggal"] = pd.to_datetime(
            df["Tahun"].astype(str) + "-" + df["Bulan"].astype(str) + "-01"
        )

materials = stok_df["Nama Bahan Baku"].dropna().unique().tolist()

selected_material = st.sidebar.selectbox(
    "Pilih Nama Bahan Baku",
    materials
)

selected_gudang = st.sidebar.selectbox(
    "Pilih Gudang",
    ["(Semua)"] + sorted(stok_df["Gudang"].dropna().unique().tolist())
)

df_filtered = stok_df[stok_df["Nama Bahan Baku"] == selected_material].copy()
if selected_gudang != "(Semua)":
    df_filtered = df_filtered[df_filtered["Gudang"] == selected_gudang]

df_filtered = df_filtered.sort_values("Tanggal")

monthly_usage = (
    df_filtered
    .groupby("Tanggal")["Usage (kg)"]
    .sum()
    .asfreq("MS")
)

if monthly_usage.isna().all():
    st.warning("Data usage tidak tersedia.")
    st.stop()

st.subheader(f"Usage Bulanan: {selected_material}")
st.line_chart(monthly_usage)

# =========================
# Sidebar Parameter
# =========================
forecast_steps = st.sidebar.number_input(
    "Periode Forecast (bulan)", min_value=3, max_value=24, value=12
)

seasonal_period = st.sidebar.number_input(
    "Periode Musiman", min_value=2, value=12
)

hw_type = st.sidebar.selectbox(
    "Tipe Holt–Winters",
    ["Additive", "Multiplicative"]
)

lead_time_days = st.sidebar.number_input(
    "Lead Time (hari)", min_value=1, value=14
)

service_level_z = st.sidebar.number_input(
    "Z-Value (Service Level)", min_value=0.0, value=1.65
)

ordering_cost = st.sidebar.number_input(
    "Biaya Pemesanan (Rp)", min_value=0.0, value=500000.0, step=10000.0
)

holding_cost_rate = st.sidebar.number_input(
    "Biaya Penyimpanan per Tahun (%)", min_value=0.0, value=0.20
)

unit_cost = st.sidebar.number_input(
    "Harga per kg (Rp)", min_value=0.0, value=10000.0
)

holding_cost = holding_cost_rate * unit_cost

# =========================
# Holt-Winters Forecasting
# =========================
if st.button("Jalankan Forecast Holt–Winters"):
    try:
        seasonal = "add" if hw_type == "Additive" else "mul"

        model = ExponentialSmoothing(
            monthly_usage,
            trend="add",
            seasonal=seasonal,
            seasonal_periods=seasonal_period
        )

        hw_fit = model.fit()
        forecast = hw_fit.forecast(forecast_steps)

        # =========================
        # Plot
        # =========================
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(monthly_usage.index, monthly_usage.values, label="Actual")
        ax.plot(forecast.index, forecast.values, label="Forecast", linestyle="--")
        ax.legend()
        ax.set_title(f"Holt–Winters Forecast: {selected_material}")
        st.pyplot(fig)

        # =========================
        # Inventory Optimization
        # =========================
        mean_demand = monthly_usage.mean()
        daily_demand = mean_demand / 30
        std_demand = monthly_usage.std() / 30

        safety_stock = service_level_z * std_demand * np.sqrt(lead_time_days)
        reorder_point = daily_demand * lead_time_days + safety_stock
        annual_demand = mean_demand * 12

        if holding_cost > 0:
            eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
        else:
            eoq = np.nan

        st.subheader("📦 Rekomendasi Pengendalian Persediaan")

        df_rec = pd.DataFrame({
            "Indikator": [
                "Rata-rata Permintaan Bulanan",
                "Safety Stock",
                "Reorder Point",
                "EOQ"
            ],
            "Nilai (kg)": [
                mean_demand,
                safety_stock,
                reorder_point,
                eoq
            ]
        })

        st.dataframe(df_rec.style.format({"Nilai (kg)": "{:.2f}"}))

        # =========================
        # Akurasi In-Sample
        # =========================
        fitted = hw_fit.fittedvalues
        mape = np.mean(
            np.abs((monthly_usage - fitted) / monthly_usage.replace(0, np.nan))
        ) * 100
        rmse = np.sqrt(np.mean((monthly_usage - fitted) ** 2))

        st.subheader("📊 Akurasi Model Holt–Winters (In-Sample)")
        st.write(f"MAPE: **{mape:.2f}%**")
        st.write(f"RMSE: **{rmse:.2f}**")

    except Exception as e:
        st.error(f"Gagal menjalankan Holt–Winters: {e}")

st.markdown("---")
st.caption("© 2025 DSS Forecasting — Implementasi Holt–Winters untuk Optimasi Persediaan")
