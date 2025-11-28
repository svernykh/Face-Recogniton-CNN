import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
# Import library deep learning kamu (contoh pakai TensorFlow/Keras)
# from tensorflow.keras.models import load_model 

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Absensi Face Recognition", layout="centered")
st.title("📸 Absensi Kelas Otomatis")

# 2. Load Model (Gunakan cache agar tidak loading terus menerus)
@st.cache_resource
def load_my_model():
    # Ganti dengan kode load model kamu yang sebenarnya
    # model = load_model('model_cnn.h5')
    # return model
    return "Model Placeholder" # Hapus baris ini jika model sudah ada

model = load_my_model()

# 3. Fungsi Preprocessing & Prediksi
def predict_face(image):
    # Convert dari format PIL ke OpenCV
    img_array = np.array(image.convert('RGB'))
    
    # (Opsional tapi PENTING) Deteksi wajah dulu pakai Haar Cascade
    # Agar yang di-feed ke CNN cuma bagian wajah, bukan background tembok
    # gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # for (x, y, w, h) in faces:
    #     img_array = img_array[y:y+h, x:x+w] # Crop wajah

    # Resize sesuai input shape model kamu (misal 128x128)
    img_resized = cv2.resize(img_array, (128, 128))
    
    # Normalize (misal bagi 255)
    img_normalized = img_resized / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)
    
    # Prediksi
    # prediction = model.predict(img_expanded)
    # class_index = np.argmax(prediction)
    # labels = ['Andi', 'Budi', 'Citra'] # Sesuaikan dengan label kelasmu
    # return labels[class_index]
    
    return "Nama Teman" # Dummy return

# 4. Fitur Kamera
img_file_buffer = st.camera_input("Ambil Foto untuk Absen")

if img_file_buffer is not None:
    # Tampilkan foto
    image = Image.open(img_file_buffer)
    
    # Lakukan Prediksi
    nama_terdeteksi = predict_face(image)
    
    st.success(f"Wajah dikenali sebagai: **{nama_terdeteksi}**")
    
    # 5. Simpan ke CSV (Log Absen)
    if st.button("Simpan Absen"):
        waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df_new = pd.DataFrame({'Nama': [nama_terdeteksi], 'Waktu': [waktu]})
        
        # Load CSV lama atau buat baru
        try:
            df_old = pd.read_csv("attendance.csv")
            df = pd.concat([df_old, df_new], ignore_index=True)
        except FileNotFoundError:
            df = df_new
            
        df.to_csv("attendance.csv", index=False)
        st.toast("Absensi berhasil disimpan!")

# 6. Tampilkan Data Absen
st.write("---")
st.subheader("📝 Log Absensi Hari Ini")
try:
    df_show = pd.read_csv("attendance.csv")
    st.dataframe(df_show.tail(5)) # Tampilkan 5 terakhir
except FileNotFoundError:
    st.write("Belum ada data absensi.")