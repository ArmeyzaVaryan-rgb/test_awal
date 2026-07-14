import pandas as pd
import re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ==========================================
# 1. KONFIGURASI HALAMAN UTAMA STREAMLIT
# ==========================================
st.set_page_config(page_title="Deteksi Komentar Judol", page_icon="🛡️", layout="centered")

st.title("🛡️ Sistem Deteksi Komentar Judi Online")
st.write("Aplikasi untuk mendeteksi apakah suatu komentar di media sosial terindikasi sebagai spam judi online atau tidak.")
st.write("\nAplikasi ini dibuat oleh Armeyza Varyan sebagai Alat test Machine Learning model logistik regresi")
st.markdown("---")

# Fungsi preprocessing bawaan Anda (bisa dikembangkan lagi isinya nanti)
def TambahanKata(teks):
    # Jika teks kosong atau null, kembalikan string kosong agar tidak error
    if pd.isna(teks):
        return ""
    return str(teks)

# ==========================================
# 2. PROSES BELAJAR MODEL (MEMAKAI CACHE)
# ==========================================
# Fungsi ini dibungkus cache agar training hanya berjalan sekali saat web pertama dibuka
@st.cache_resource
def latih_model_sistem():
    # Membaca data
    df = pd.read_csv('Data_Komentar.csv')
    
    # Membersihkan data dari nilai kosong (null) agar TfidfVectorizer tidak error
    df = df.dropna(subset=['komentar_clean'])
    
    df_clear = df[['komentar_clean','predicted_label']]
    
    # Menerapkan fungsi preprocessing
    df['komentar_bersih'] = df_clear['komentar_clean'].apply(TambahanKata)
    
    X_clean = df['komentar_bersih']
    Y_clean = df_clear['predicted_label']
    
    # Memisahkan data train dan test (80:20)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_clean, Y_clean, 
        test_size=0.2, 
        random_state=42, 
        stratify=Y_clean
    )
    
    # Inisialisasi dan fitting vectorizer
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Melatih model
    model = LogisticRegression()
    model.fit(X_train_vec, Y_train)
    
    # Menghitung akurasi untuk performa sistem
    y_prediksi = model.predict(X_test_vec)
    skor_akurasi = accuracy_score(Y_test, y_prediksi)
    
    return model, vectorizer, skor_akurasi

# Panggil fungsi latihan model di atas
with st.spinner('Sedang memuat sistem dan melatih model, mohon tunggu...'):
    model, vectorizer, akurasi_sistem = latih_model_sistem()

# Menampilkan informasi performa model di sidebar (samping kiri web)
st.sidebar.header("📊 Performa Model")
st.sidebar.metric(label="Akurasi Ujian Sistem", value=f"{akurasi_sistem * 100:.2f}%")
st.sidebar.write("Model dilatih menggunakan algoritma *Logistic Regression* dan *TF-IDF Vectorizer*.")

# ==========================================
# 3. ANTARMUKA PENGGUNA (INPUT & PREDIKSI)
# ==========================================
st.subheader("🔮 Uji Coba Prediksi Komentar Baru")

# Mengganti fungsi input() bawaan python dengan text_area milik streamlit
data_mentah = st.text_area(
    label="Masukkan teks komentar yang ingin dianalisis:", 
    placeholder="Contoh: Info situs g4c0r malam ini kak langsung jp..."
)

# Membuat tombol untuk mengeksekusi prediksi
if st.button("Analisis Komentar", type="primary"):
    if data_mentah.strip() == "":
        st.warning("⚠️ Mohon ketikkan teks komentar terlebih dahulu sebelum menekan tombol!")
    else:
        # Jalur pemrosesan teks baru sesuai logika Anda
        data_bersih = TambahanKata(data_mentah)
        data_angka = vectorizer.transform([data_bersih])
        
        # Prediksi hasil tebakan
        hasil_tebakan = model.predict(data_angka)
        
        # Menampilkan output hasil tebakan dengan kotak visual Streamlit
        st.markdown("### 📢 Hasil Analisis:")
        if hasil_tebakan[0] == 0:
            st.success("✅ **Hasil Tebakan:** BUKAN JUDOL (Komentar Aman / Normal)")
        elif hasil_tebakan[0] == 1:
            st.error("❌ **Hasil Tebakan:** TERINDIKASI JUDOL (Wajib Diblokir / Spam)")
