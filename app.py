import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import datetime
import json
# from lucide_react import Plane, Clock, AlertTriangle, Table  <- DIHAPUS, KARENA INI REACT

# --- (BARU) URL Gambar Latar ---
BACKGROUND_IMAGE_URL = "https://images.unsplash.com/photo-1569154941061-623bc2b11c34?auto=format&fit=crop&w=1920&q=80"

# --- 1. Konfigurasi Halaman ---
st.set_page_config(page_title="Prediksi Keterlambatan Penerbangan", layout="wide")

# --- (BARU) Terapkan CSS untuk Background ---
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)), url("{BACKGROUND_IMAGE_URL}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .stApp > header {{
        background-color: transparent;
    }}
    div[data-testid="stAppViewContainer"] > .main > div:first-child {{
        padding-top: 3rem;
    }}
    /* (DIPERBARUI) Target yang lebih spesifik untuk kartu konten */
    section[data-testid="st.main"] > div:first-child > div:first-child > div:first-child {{
        background-color: rgba(255, 255, 255, 0.95); /* Kartu putih semi-transparan */
        border-radius: 10px;
        padding: 2rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# --- 2. Muat Model dan Data Pendukung ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('flight_delay_model.joblib')
        weather_daily = pd.read_csv('weather_daily_processed.csv')
        model_columns = joblib.load('model_columns.joblib')
        categorical_cols = joblib.load('categorical_features.joblib')
        
        with open('lookup_maps.json', 'r') as f:
            lookup_maps = json.load(f)
        
        holiday_dates = set(lookup_maps['holiday_dates'])
        holiday_window = set(lookup_maps['holiday_window'])
        
        weather_daily['merge_key_date'] = pd.to_datetime(weather_daily['merge_key_date']).dt.date
        
        # Ekstrak opsi dropdown dari lookup_maps
        airline_options = sorted(list(lookup_maps['airline_map'].keys()))
        airport_options = sorted(list(lookup_maps['airport_to_city_map'].keys()))
        
        return model, weather_daily, model_columns, categorical_cols, lookup_maps, holiday_dates, holiday_window, airline_options, airport_options
    
    except FileNotFoundError as e:
        st.error(f"ERROR: File aset tidak ditemukan. Pastikan semua file (.joblib, .csv, .json) ada di repository. Error: {e}")
        return (None,) * 9

(model, weather_daily, model_columns, categorical_cols, 
 lookup_maps, holiday_dates, holiday_window, 
 airline_options, airport_options) = load_assets()

# --- Fungsi Helper ---
def get_deptime_label(hour):
    if 5 <= hour < 12: return "Morning"
    elif 12 <= hour < 17: return "Afternoon"
    elif 17 <= hour < 21: return "Evening"
    else: return "Night"

# --- Komponen Tabel Fitur (diubah untuk Streamlit) ---
def FeatureTable(data):
    featureNameMap = {
        'Day_Of_Week': 'Hari dalam Minggu (1=Senin, 7=Minggu)', 'Airline': 'Maskapai',
        'Dep_Airport': 'Bandara Keberangkatan (Kode)', 'Dep_CityName': 'Kota Keberangkatan',
        'DepTime_label': 'Waktu Keberangkatan', 'Dep_Delay': 'Keterlambatan Berangkat (Menit)',
        'Arr_Airport': 'Bandara Tujuan (Kode)', 'Arr_CityName': 'Kota Tujuan',
        'Flight_Duration': 'Durasi Penerbangan (Menit)', 'Distance_type': 'Tipe Jarak',
        'Manufacturer': 'Pabrikan Pesawat', 'Model': 'Model Pesawat',
        'Aicraft_age': 'Usia Pesawat (Tahun)', 'Is_Holiday': 'Status Hari Libur (1=Ya)',
        'Is_Near_Holiday': 'Status Dekat Libur (1=Ya)', 'Delay_NAS': 'Input Delay NAS (Menit)',
        'Delay_LastAircraft': 'Input Delay Pesawat Sblmnya (Menit)',
        'origin_tavg': 'Suhu Rata-Rata (Asal °C)', 'origin_prcp': 'Curah Hujan (Asal mm)',
        'origin_wspd': 'Kecepatan Angin (Asal km/jam)', 'dest_tavg': 'Suhu Rata-Rata (Tujuan °C)',
        'dest_prcp': 'Curah Hujan (Tujuan mm)', 'dest_wspd': 'Kecepatan Angin (Tujuan km/jam)'
    }
    
    # Filter data yang relevan (menyembunyikan data cuaca yang lebih teknis)
    relevant_data = {
        featureNameMap.get(key, key): val 
        for key, val in data.items() 
        if key in featureNameMap and (val != 0 and val != 'Unknown' and val is not None and val != "")
    }
    
    st.table(pd.DataFrame(list(relevant_data.items()), columns=['Parameter Prediksi', 'Nilai yang Digunakan']))


# --- 3. UI (Form Input) ---
if model is not None:
    # Buat satu blok vertikal utama untuk konten
    with st.container():
        st.title('✈️ Prediktor Keterlambatan Penerbangan')
        st.write("Aplikasi ini memprediksi apakah penerbangan Anda akan terlambat lebih dari 15 menit.")
        
        st.subheader("Masukkan Detail Penerbangan:")
        
        col1, col2 = st.columns(2)
        with col1:
            flight_date_input = st.date_input("Tanggal Penerbangan", datetime.date(2023, 1, 15))
            time_input = st.time_input("Jam Keberangkatan Terjadwal", datetime.time(9, 30))
            # (DIPERBARUI) Menggunakan opsi yang dimuat dari JSON
            airline_input = st.selectbox("Maskapai (Airline)", airline_options, index=airline_options.index('Endeavor Air') if 'Endeavor Air' in airline_options else 0)
            dep_delay_input = st.number_input("Keterlambatan Berangkat (Menit)", min_value=-60, max_value=600, value=0, help="Masukkan 0 jika tepat waktu, atau angka negatif jika berangkat lebih awal.")

        with col2:
            # (DIPERBARUI) Menggunakan opsi yang dimuat dari JSON
            dep_airport_input = st.selectbox("Bandara Keberangkatan (Origin)", airport_options, index=airport_options.index('ATL') if 'ATL' in airport_options else 0)
            arr_airport_input = st.selectbox("Bandara Kedatangan (Destination)", airport_options, index=airport_options.index('CVG') if 'CVG' in airport_options else 0)
            duration_input = st.number_input("Durasi Penerbangan (Menit)", min_value=30, max_value=600, value=120)
        
        st.subheader("Info Tambahan (Opsional, namun sangat mempengaruhi prediksi):")
        col3, col4 = st.columns(2)
        with col3:
             delay_nas_input = st.number_input("Keterlambatan Lalu Lintas Udara (Delay_NAS)", min_value=0, max_value=300, value=0)
        with col4:
            delay_last_input = st.number_input("Keterlambatan Pesawat Sblmnya (Delay_LastAircraft)", min_value=0, max_value=300, value=0)

        st.markdown("---")

        # --- 4. Tombol Prediksi dan Logika Backend ---
        if st.button('🚀 Prediksi Keterlambatan', use_container_width=True, type="primary"):
            
            with st.spinner('Menganalisis data cuaca dan lalu lintas...'):
                merge_key_date_obj = flight_date_input
                merge_key_date_str = merge_key_date_obj.isoformat()

                # --- 5. Kalkulasi ETA ---
                scheduled_departure_dt = datetime.datetime.combine(flight_date_input, time_input)
                total_known_delay_min = dep_delay_input + delay_nas_input + delay_last_input
                actual_departure_dt = scheduled_departure_dt + datetime.timedelta(minutes=total_known_delay_min)
                estimated_arrival_dt = actual_departure_dt + datetime.timedelta(minutes=duration_input)
                eta_time_str = estimated_arrival_dt.strftime("%H:%M")
                eta_date_str = estimated_arrival_dt.strftime("%Y-%m-%d")
                is_next_day = (estimated_arrival_dt.date() > scheduled_departure_dt.date())
                eta_display = f"{eta_time_str} pada {eta_date_str}" + (" (hari berikutnya)" if is_next_day else "")
                
                # --- 6. Feature Engineering (Cuaca & Kalender) ---
                origin_weather = weather_daily[(weather_daily['airport_id'] == dep_airport_input) & (weather_daily['merge_key_date'] == merge_key_date_obj)].copy()
                origin_rename = {col: f"origin_{col}" for col in weather_daily.columns if col not in ['airport_id', 'merge_key_date']}
                origin_weather = origin_weather.rename(columns=origin_rename)
                
                dest_weather = weather_daily[(weather_daily['airport_id'] == arr_airport_input) & (weather_daily['merge_key_date'] == merge_key_date_obj)].copy()
                dest_rename = {col: f"dest_{col}" for col in weather_daily.columns if col not in ['airport_id', 'merge_key_date']}
                dest_weather = dest_weather.rename(columns=dest_rename)

                airport_to_city_map = lookup_maps['airport_to_city_map']
                airline_map = lookup_maps['airline_map']
                default_values = lookup_maps['default_values']
                airline_info = airline_map.get(airline_input, {})

                is_holiday_flag = 1 if merge_key_date_str in holiday_dates else 0
                is_near_holiday_flag = 1 if merge_key_date_str in holiday_window else 0

                input_data = {
                    'Airline': airline_input, 'Dep_Airport': dep_airport_input, 'Arr_Airport': arr_airport_input,
                    'Dep_Delay': dep_delay_input, 'Flight_Duration': duration_input,
                    'Day_Of_Week': flight_date_input.weekday() + 1,
                    'Delay_NAS': delay_nas_input, 'Delay_LastAircraft': delay_last_input,
                    'Dep_CityName': airport_to_city_map.get(dep_airport_input, 'Unknown'),
                    'Arr_CityName': airport_to_city_map.get(arr_airport_input, 'Unknown'),
                    'DepTime_label': get_deptime_label(time_input.hour), 
                    'Is_Holiday': is_holiday_flag, 'Is_Near_Holiday': is_near_holiday_flag,
                    'Distance_type': default_values.get('Distance_type', 'Unknown'),
                    'Manufacturer': airline_info.get('Manufacturer', default_values.get('Manufacturer', 'Unknown')),
                    'Model': airline_info.get('Model', default_values.get('Model', 'Unknown')),
                    'Aicraft_age': airline_info.get('Aicraft_age', default_values.get('Aicraft_age', 0))
                }
                
                input_df = pd.DataFrame([input_data])
                input_df = pd.concat([
                    input_df.reset_index(drop=True),
                    origin_weather.reset_index(drop=True).drop(columns=['airport_id', 'merge_key_date'], errors='ignore'),
                    dest_weather.reset_index(drop=True).drop(columns=['airport_id', 'merge_key_date'], errors='ignore')
                ], axis=1)
                
                input_df = input_df.fillna(0)
                input_df = input_df.reindex(columns=model_columns, fill_value=0)

                for col in categorical_cols:
                    if col in input_df.columns:
                        input_df[col] = input_df[col].astype('category')
                
                # --- 7. Prediksi ---
                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0][1]

                # --- 8. Tampilkan Hasil ---
                st.subheader("Hasil Prediksi:")
                
                if is_near_holiday_flag == 1:
                    st.warning("⚠️ Catatan: Prediksi ini memperhitungkan kepadatan hari libur.", icon="⚠️") # Emoji Icon

                res_col1, res_col2 = st.columns(2)
                if prediction == 1:
                    with res_col1:
                        # (DIPERBARUI) Menggunakan emoji, bukan komponen Lucide
                        st.error(f"**Penerbangan Diprediksi TERLAMBAT**\n\nProbabilitas Keterlambatan: **{(probability * 100):.0f}%**", icon="⚠️")
                else:
                    with res_col1:
                        # (DIPERBARUI) Menggunakan emoji, bukan komponen Lucide
                        st.success(f"**Penerbangan Diprediksi TEPAT WAKTU**\n\nProbabilitas Tepat Waktu: **{(100 - (probability * 100)):.0f}%**", icon="✈️")
                
                with res_col2:
                    # (DIPERBARUI) Menggunakan emoji, bukan komponen Lucide
                    st.info(f"**Estimasi Waktu Kedatangan (ETA)**\n\n**{eta_display}**", icon="🕒")
                    if prediction == 1:
                        st.warning("ETA ini mungkin meleset karena adanya prediksi *tambahan* delay.", icon="⚠️")
                
                # Tampilkan tabel rincian (selalu ditampilkan)
                with st.expander("Tampilkan Rincian Data Prediksi", expanded=True): # 'expanded=True' agar terbuka otomatis
                    FeatureTable(input_data)
else:
    st.error("Gagal memuat aset model. Silakan periksa file di repository dan coba lagi.")
