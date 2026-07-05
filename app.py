# DUA BARIS INI WAJIB DI PALING ATAS
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model 
from tensorflow.keras.layers import DepthwiseConv2D 
import sqlite3
import pandas as pd
from datetime import datetime

# --- TRIK UNTUK MENGATASI ERROR TEACHABLE MACHINE ---
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        if 'groups' in kwargs:
            del kwargs['groups'] 
        super().__init__(**kwargs)
# ----------------------------------------------------

# --- FUNGSI DATABASE SQLITE ---
def init_db():
    conn = sqlite3.connect('absensi.db')
    c = conn.cursor()
    # Tabel untuk riwayat absensi dari kamera
    c.execute('''CREATE TABLE IF NOT EXISTS riwayat
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  nama TEXT,
                  waktu TEXT,
                  akurasi TEXT)''')
    # Tabel baru untuk kelengkapan administrasi (CRUD)
    c.execute('''CREATE TABLE IF NOT EXISTS pengguna
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  nim TEXT,
                  nama TEXT,
                  konsentrasi TEXT)''')
    conn.commit()
    conn.close()

def catat_absen(nama, akurasi):
    conn = sqlite3.connect('absensi.db')
    c = conn.cursor()
    waktu_sekarang = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO riwayat (nama, waktu, akurasi) VALUES (?, ?, ?)", (nama, waktu_sekarang, akurasi))
    conn.commit()
    conn.close()

# Fungsi CRUD Pengguna
def tambah_pengguna(nim, nama, konsentrasi):
    conn = sqlite3.connect('absensi.db')
    c = conn.cursor()
    c.execute("INSERT INTO pengguna (nim, nama, konsentrasi) VALUES (?, ?, ?)", (nim, nama, konsentrasi))
    conn.commit()
    conn.close()

def hapus_pengguna(nim):
    conn = sqlite3.connect('absensi.db')
    c = conn.cursor()
    c.execute("DELETE FROM pengguna WHERE nim=?", (nim,))
    conn.commit()
    conn.close()
# ------------------------------

# 1. KONFIGURASI HALAMAN & INIT DB
st.set_page_config(page_title="Sistem Absensi Wajah", layout="wide")
init_db() 

# 2. FUNGSI UNTUK LOAD MODEL TEACHABLE MACHINE
@st.cache_resource
def load_teachable_machine_model():
    model = load_model("keras_model.h5", compile=False, custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})
    class_names = open("labels.txt", "r").readlines()
    return model, class_names

# 3. MEMBUAT MENU NAVIGASI
st.sidebar.title("Navigasi")
menu = st.sidebar.selectbox("Pilih Halaman:", ["Absensi Wajah", "Login Admin"])

# ==========================================
# HALAMAN 1: ABSENSI WAJAH (Frontend Pengguna)
# ==========================================
if menu == "Absensi Wajah":
    st.title("📷 Sistem Absensi Menggunakan Wajah")
    st.write("Silakan menghadap ke kamera dan ambil foto untuk absensi.")
    
    model, class_names = load_teachable_machine_model()
    gambar_kamera = st.camera_input("Ambil Foto")

    if gambar_kamera is not None:
        image = Image.open(gambar_kamera)
        image = image.resize((224, 224)) 
        image_array = np.asarray(image)

        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        st.info("Memproses wajah...")

        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name_raw = class_names[index].strip()
        confidence_score = prediction[0][index]
        
        nama_terdeteksi = class_name_raw[2:]
        akurasi_persen = np.round(confidence_score * 100)

        st.write(f"**Wajah Terdeteksi:** {nama_terdeteksi}") 
        st.write(f"**Tingkat Akurasi:** {akurasi_persen}%")
        
        if akurasi_persen > 70 and "background" not in nama_terdeteksi.lower() and "kosong" not in nama_terdeteksi.lower():
            catat_absen(nama_terdeteksi, f"{akurasi_persen}%")
            st.success(f"✅ Berhasil! Kehadiran {nama_terdeteksi} telah dicatat ke database.")
        elif "background" in nama_terdeteksi.lower() or "kosong" in nama_terdeteksi.lower():
            st.warning("⚠️ Wajah tidak ditemukan. Pastikan wajahmu terlihat jelas di kamera.")
        else:
            st.error("❌ Wajah terdeteksi tapi akurasi rendah. Coba foto ulang.")

# ==========================================
# HALAMAN 2: DASHBOARD ADMIN & CRUD
# ==========================================
elif menu == "Login Admin":
    st.title("🔐 Login Administrator")
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state['login_berhasil'] = True
            st.success("Login berhasil! Silakan buka Dashboard.")
        else:
            st.error("Username atau password salah!")

    if 'login_berhasil' in st.session_state and st.session_state['login_berhasil']:
        st.markdown("---")
        st.header("📊 Dashboard Admin")
        
        tab1, tab2, tab3 = st.tabs(["Riwayat Kehadiran", "Data Pengguna (CRUD)", "Status Model AI"])
        
        with tab1:
            st.subheader("Riwayat Kehadiran Hari Ini")
            conn = sqlite3.connect('absensi.db')
            df_riwayat = pd.read_sql_query("SELECT * FROM riwayat ORDER BY id DESC", conn)
            conn.close()
            
            if df_riwayat.empty:
                st.write("Belum ada data absensi masuk.")
            else:
                st.dataframe(df_riwayat, use_container_width=True, hide_index=True)
            
        with tab2:
            st.subheader("Manajemen Data Pengguna")
            st.info("Catatan: Penambahan data di sini adalah untuk kelengkapan administrasi. Agar sistem mengenali wajah, data harus di-training di Teachable Machine terlebih dahulu.")
            
            # -- FORM TAMBAH DATA --
            with st.expander("➕ Tambah Pengguna Baru"):
                with st.form("form_tambah"):
                    nim_baru = st.text_input("NIM")
                    nama_baru = st.text_input("Nama Lengkap", placeholder="Contoh: Ghulam Yahya")
                    konsentrasi_baru = st.selectbox("Konsentrasi", ["Multimedia", "Jaringan", "Pemrograman"])
                    submit_tambah = st.form_submit_button("Simpan Data")
                    
                    if submit_tambah:
                        if nim_baru and nama_baru:
                            tambah_pengguna(nim_baru, nama_baru, konsentrasi_baru)
                            st.success(f"Data {nama_baru} berhasil ditambahkan!")
                            st.rerun() # Refresh halaman untuk update tabel
                        else:
                            st.error("NIM dan Nama wajib diisi!")
            
            # -- TABEL DATA PENGGUNA --
            st.write("**Daftar Pengguna Terdaftar:**")
            conn = sqlite3.connect('absensi.db')
            df_pengguna = pd.read_sql_query("SELECT nim AS NIM, nama AS Nama, konsentrasi AS Konsentrasi FROM pengguna", conn)
            conn.close()
            
            if df_pengguna.empty:
                st.write("Belum ada data pengguna.")
            else:
                st.dataframe(df_pengguna, use_container_width=True, hide_index=True)
                
            # -- FORM HAPUS DATA --
            with st.expander("🗑️ Hapus Pengguna"):
                with st.form("form_hapus"):
                    nim_hapus = st.text_input("Masukkan NIM yang akan dihapus")
                    submit_hapus = st.form_submit_button("Hapus Data")
                    
                    if submit_hapus:
                        if nim_hapus:
                            hapus_pengguna(nim_hapus)
                            st.success(f"Data dengan NIM {nim_hapus} berhasil dihapus!")
                            st.rerun()
                        else:
                            st.error("NIM wajib diisi!")
            
        with tab3:
            st.subheader("Status Sistem")
            st.write("Database: **Terkoneksi (SQLite)**")
            st.write("Model AI: **Aktif & Siap digunakan**")