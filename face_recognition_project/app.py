import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import os

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Absensi Face Recognition", layout="centered")
st.title("📸 Absensi Kelas Otomatis")

# Konfigurasi Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Load Label Map
@st.cache_data
def load_label_map():
    # Gunakan path relative terhadap lokasi file app.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # dataset.csv ada di parent directory (root project)
    csv_path = os.path.join(os.path.dirname(current_dir), "dataset.csv")
    
    if not os.path.exists(csv_path):
        st.error(f"File {csv_path} tidak ditemukan! Pastikan path dataset benar.")
        return {}
    
    df = pd.read_csv(csv_path)
    
    # Kita butuh idx -> label untuk prediksi
    unique_labels = df['label'].unique()
    idx_to_label = {idx: label for idx, label in enumerate(unique_labels)}
    return idx_to_label

idx_to_label = load_label_map()
num_classes = len(idx_to_label)

# 3. Load Model
def get_model(num_classes):
    # Load Pretrained EfficientNet V2 M structure
    weights = models.EfficientNet_V2_M_Weights.DEFAULT
    model = models.efficientnet_v2_m(weights=weights)
    
    # Replicate the modification logic
    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze the last two feature blocks
    for param in model.features[-1].parameters():
        param.requires_grad = True
    for param in model.features[-2].parameters():
        param.requires_grad = True
        
    # Replace Classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    
    return model

@st.cache_resource
def load_my_model(num_classes):
    model = get_model(num_classes)
    
    # Load weights relative to app.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "efficientnet_v2_m_finetuned.pth")
    
    if not os.path.exists(model_path):
        st.error(f"Model {model_path} tidak ditemukan!")
        return None
    
    # Load state dict
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

if num_classes > 0:
    model = load_my_model(num_classes)
else:
    st.error("Gagal memuat label map. Aplikasi tidak dapat berjalan.")
    model = None

# Preprocessing transforms
IMG_SIZE = 480
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 4. Fungsi Preprocessing & Prediksi
def predict_face(image):
    if model is None:
        return "Model Error"
    
    # Convert PIL to OpenCV format for Haar Cascade
    img_array = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Load Haar Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        return "Wajah tidak terdeteksi"
    
    # Ambil wajah pertama yang terdeteksi
    (x, y, w, h) = faces[0]
    
    # Crop wajah
    face_img = image.crop((x, y, x+w, y+h))
    
    # Transform untuk model
    img_tensor = val_transform(face_img)
    img_expanded = img_tensor.unsqueeze(0).to(device) # Add batch dimension
    
    # Prediksi
    with torch.no_grad():
        outputs = model(img_expanded)
        _, predicted = outputs.max(1)
        class_index = predicted.item()
        
    return idx_to_label.get(class_index, "Unknown")

# 5. Fitur Kamera
img_file_buffer = st.camera_input("Ambil Foto untuk Absen")

if img_file_buffer is not None:
    # Tampilkan foto
    image = Image.open(img_file_buffer)
    
    # Lakukan Prediksi
    nama_terdeteksi = predict_face(image)
    
    if nama_terdeteksi in ["Wajah tidak terdeteksi", "Model Error", "Unknown"]:
        st.warning(f"Status: {nama_terdeteksi}")
    else:
        st.success(f"Wajah dikenali sebagai: **{nama_terdeteksi}**")
        
        # 6. Simpan ke CSV (Log Absen)
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

# 7. Tampilkan Data Absen
st.write("---")
st.subheader("📝 Log Absensi Hari Ini")
try:
    df_show = pd.read_csv("attendance.csv")
    st.dataframe(df_show.tail(5)) # Tampilkan 5 terakhir
except FileNotFoundError:
    st.write("Belum ada data absensi.")