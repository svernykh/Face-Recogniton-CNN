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
import urllib.request
import mediapipe as mp

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Absensi Face Recognition (Swin)", layout="centered")
st.title("📸 Absensi Kelas Otomatis (Swin Transformer)")

# Konfigurasi Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# 2. Load Label Map
@st.cache_data
def load_label_map():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(os.path.dirname(current_dir), "dataset.csv")
    
    if not os.path.exists(csv_path):
        st.error(f"File {csv_path} tidak ditemukan! Pastikan path dataset benar.")
        return {}
    
    df = pd.read_csv(csv_path)
    unique_labels = df['label'].unique()
    idx_to_label = {idx: label for idx, label in enumerate(unique_labels)}
    return idx_to_label

idx_to_label = load_label_map()
num_classes = len(idx_to_label)

# 3. Load Model
def get_model(num_classes):
    # Load Pretrained Swin Transformer V2 Tiny
    weights = models.Swin_V2_T_Weights.IMAGENET1K_V1
    model = models.swin_v2_t(weights=weights)
    
    # Freeze semua layer
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze layer terakhir dari features
    for param in model.features[-1].parameters():
        param.requires_grad = True
    for param in model.features[-2].parameters():
        param.requires_grad = True
        
    # Replace Head (Classifier)
    in_features = model.head.in_features
    model.head = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    
    return model

@st.cache_resource
def load_my_model(num_classes):
    model = get_model(num_classes)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "swin_v2_t_finetuned.pth")
    
    if not os.path.exists(model_path):
        st.error(f"Model {model_path} tidak ditemukan!")
        return None
    
    # Use weights_only=True if possible, but for full state dict usually safe with our own model
    # Suppress warning by not using it or ignoring
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

if num_classes > 0:
    model = load_my_model(num_classes)
else:
    st.error("Gagal memuat label map. Aplikasi tidak dapat berjalan.")
    model = None

# Preprocessing
IMG_SIZE = 256 # Swin Transformer V2 T
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 4. Fungsi Preprocessing & Prediksi
def predict_face(image):
    if model is None:
        return None, "Model Error", 0.0, None
    
    # Convert PIL Image to NumPy array (RGB)
    img_array = np.array(image.convert('RGB'))
    
    # MediaPipe process
    results = face_detection.process(img_array)
    
    img_with_box = img_array.copy()
    
    if not results.detections:
        return img_with_box, "Wajah tidak terdeteksi", 0.0, None
        
    # Ambil deteksi dengan score tertinggi
    best_detection = max(results.detections, key=lambda d: d.score[0])
    
    # Extract Bounding Box
    bboxC = best_detection.location_data.relative_bounding_box
    ih, iw, _ = img_array.shape
    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

    # Expand bounding box to include whole head/hair
    pad_top = int(h * 0.5)    # Add 50% height to top
    pad_bottom = int(h * 0.1) # Add 10% height to bottom
    pad_side = int(w * 0.2)   # Add 20% width to sides

    x = x - pad_side
    y = y - pad_top
    w = w + (pad_side * 2)
    h = h + pad_top + pad_bottom
    
    # Ensure box is within image bounds
    x, y = max(0, x), max(0, y)
    w, h = min(w, iw - x), min(h, ih - y)
    
    # Draw Box
    cv2.rectangle(img_with_box, (x, y), (x + w, y + h), (0, 255, 0), 3)
    
    # Crop Face
    face_img = image.crop((x, y, x + w, y + h))
    
    # Transform for Model
    try:
        img_tensor = val_transform(face_img)
        img_expanded = img_tensor.unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(img_expanded)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            max_prob, predicted = probabilities.max(1)
            class_index = predicted.item()
            confidence = max_prob.item()
            
        predicted_name = idx_to_label.get(class_index, "Unknown")
        return img_with_box, predicted_name, confidence, face_img
        
    except Exception as e:
        return img_with_box, f"Error: {str(e)}", 0.0, None

# 5. Fitur Kamera Real-Time
if 'run_camera' not in st.session_state:
    st.session_state.run_camera = False

# State untuk menyimpan data sementara sebelum konfirmasi
if 'pending_attendance' not in st.session_state:
    st.session_state.pending_attendance = None

# Jika ada data pending, tampilkan UI Konfirmasi
if st.session_state.pending_attendance:
    st.info("Wajah Terdeteksi! Silakan konfirmasi.")
    
    col_img, col_info = st.columns([1, 2])
    
    with col_img:
        # Tampilkan wajah yang dicrop
        st.image(st.session_state.pending_attendance['face_img'], caption="Wajah Terdeteksi", width=150)
        
    with col_info:
        st.success(f"Nama: **{st.session_state.pending_attendance['name']}**")
        st.write(f"Confidence: {st.session_state.pending_attendance['confidence']:.2%}")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("✅ Konfirmasi Absen", type="primary"):
                nama = st.session_state.pending_attendance['name']
                waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Simpan ke CSV
                df_new = pd.DataFrame({'Nama': [nama], 'Waktu': [waktu]})
                try:
                    df_old = pd.read_csv("attendance.csv")
                    df = pd.concat([df_old, df_new], ignore_index=True)
                except FileNotFoundError:
                    df = df_new
                
                df.to_csv("attendance.csv", index=False)
                
                st.toast(f"Absensi berhasil disimpan: {nama}")
                st.session_state.pending_attendance = None # Reset state
                st.rerun()
                
        with col_btn2:
            if st.button("🔄 Scan Ulang"):
                st.session_state.pending_attendance = None # Reset state
                st.session_state.run_camera = True # Nyalakan kamera lagi
                st.rerun()

else:
    # UI Kamera Normal
    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("Mulai Kamera", type="primary")
    with col2:
        stop_button = st.button("Stop Kamera", type="secondary")

    if start_button:
        st.session_state.run_camera = True
    if stop_button:
        st.session_state.run_camera = False

    # Placeholder untuk video
    frame_placeholder = st.empty()
    status_placeholder = st.empty()

    CONFIDENCE_THRESHOLD = 0.60

    if st.session_state.run_camera:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Tidak dapat membuka kamera.")
        else:
            while st.session_state.run_camera:
                ret, frame = cap.read()
                if not ret:
                    st.error("Gagal membaca frame dari kamera.")
                    break
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                
                img_boxed, nama_terdeteksi, confidence, cropped_face = predict_face(pil_image)
                
                display_frame = img_boxed.copy()
                status_color = (255, 0, 0)
                status_text = "Mencari Wajah..."
                
                if nama_terdeteksi not in ["Wajah tidak terdeteksi", "Model Error"] and not nama_terdeteksi.startswith("Error"):
                    if confidence >= CONFIDENCE_THRESHOLD:
                        status_color = (0, 255, 0)
                        status_text = f"Ditemukan: {nama_terdeteksi}"
                        
                        # --- PAUSE & CONFIRM LOGIC ---
                        # Simpan data ke session state
                        st.session_state.pending_attendance = {
                            'name': nama_terdeteksi,
                            'confidence': confidence,
                            'face_img': cropped_face
                        }
                        st.session_state.run_camera = False # Stop loop
                        cap.release()
                        st.rerun() # Refresh halaman untuk masuk ke mode konfirmasi
                        
                    else:
                        status_color = (255, 255, 0)
                        status_text = f"Tidak Yakin: {nama_terdeteksi} ({confidence:.0%})"
                    
                elif nama_terdeteksi == "Wajah tidak terdeteksi":
                     status_text = "Wajah tidak terdeteksi"
                
                cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                frame_placeholder.image(display_frame, channels="RGB", use_container_width=True)
                status_placeholder.info(f"Status: {status_text}")

            cap.release()
            frame_placeholder.empty()
            status_placeholder.empty()
            if not st.session_state.pending_attendance:
                st.write("Kamera berhenti.")

# 7. Tampilkan Data Absen
st.write("---")
st.subheader("📝 Log Absensi Hari Ini")
try:
    df_show = pd.read_csv("attendance.csv")
    # Tampilkan data terbaru di atas
    st.dataframe(df_show.iloc[::-1].head(10))
except FileNotFoundError:
    st.write("Belum ada data absensi.")
