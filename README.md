***

# Face-Recognition-CNN – EfficientNetV2-M

Proyek ini mengimplementasikan pipeline lengkap face recognition berbasis Convolutional Neural Network (CNN) menggunakan arsitektur EfficientNet V2 M dari `torchvision` sebagai backbone yang di‐fine-tune untuk tugas klasifikasi identitas wajah.[1][2]

## Fitur Utama

- Fine-tuning EfficientNet V2 M (pretrained ImageNet) untuk face recognition multi-kelas.[2][1]
- Pipeline end-to-end di satu notebook: load dataset dari CSV, augmentasi, training, evaluasi, dan visualisasi metrik.[1]
- Early stopping dengan patience 3 epoch dan penyimpanan model terbaik berdasarkan akurasi validasi.[1]
- Perhitungan akurasi dan confusion matrix untuk mengevaluasi performa model.[1]
- Struktur dataset sederhana berbasis folder per identitas + file `dataset.csv` sebagai indeks gambar.[1]

## Struktur Proyek

```bash
Face-Recogniton-CNN/
├── face_recognition_project/      # (opsional, folder tambahan dari penulis)
├── Train/                         # Folder utama data training
│   ├── person_1/
│   │   ├── img_1.jpg
│   │   └── ...
│   ├── person_2/
│   └── ...
├── dataset.csv                    # CSV: kolom minimal ['gambar', 'label']
├── attendance.csv                 # Contoh file kehadiran (opsional)
├── create_dataset_csv.py          # Script untuk membuat dataset.csv dari folder Train
├── face_recognition_pipeline.ipynb# Notebook utama pipeline training & evaluasi
├── requirements.txt               # Dependensi Python
├── .gitignore
└── .gitattributes
```

`dataset.csv` berisi minimal dua kolom:  
- `gambar`: nama file gambar (misal `img_1.jpg`).  
- `label`: nama folder/kelas (misal `person_1`) yang akan di-encode menjadi `label_idx` di notebook.[3]

## Instalasi

1. Clone repository:

```bash
git clone https://github.com/svernykh/Face-Recogniton-CNN.git
cd Face-Recogniton-CNN
```

2. Buat dan aktifkan virtual environment (opsional tapi direkomendasikan):

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

3. Install dependensi:

```bash
pip install -r requirements.txt
```

`requirements.txt` sudah mengatur extra index untuk wheel PyTorch CUDA 12.1, serta library umum seperti `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `torch`, `torchvision`, `torchaudio`, `pillow`, dan `tqdm`.[3]

## Persiapan Dataset

1. Siapkan struktur folder gambar:

```bash
Train/
├── person_1/
│   ├── img_1.jpg
│   ├── img_2.jpg
│   └── ...
├── person_2/
│   └── ...
└── ...
```

Setiap folder di dalam `Train/` merepresentasikan satu identitas (label wajah).  

2. Jalankan script untuk membuat `dataset.csv`:

```bash
python create_dataset_csv.py
```

Script ini akan memindai folder `Train/` dan menghasilkan file `dataset.csv` berisi nama file dan label yang dibutuhkan notebook.[3]

## Konfigurasi Model & Training

Parameter penting yang digunakan di notebook `face_recognition_pipeline.ipynb` antara lain:[4]

- Model: `efficientnet_v2_m` pretrained dari `torchvision.models`.  
- Fine-tuning:  
  - Semua layer dibekukan kecuali classifier dan dua layer terakhir (bagian akhir backbone).  
- Hyperparameter:
  - Epoch: `12`  
  - Batch size: `32`  
  - Learning rate: `0.001`  
  - Patience early stopping: `3`  
  - Ukuran input: `IMG_SIZE = 480` (resolusi default EfficientNetV2-M).[4]

Augmentasi dan normalisasi:

- Resize ke `(480, 480)`  
- `RandomHorizontalFlip` dan `RandomRotation(10°)` untuk data train  
- Normalisasi dengan mean `[0.485, 0.456, 0.406]` dan std `[0.229, 0.224, 0.225]` (standar ImageNet).[4]

## Cara Menjalankan Notebook

1. Buka notebook:

```bash
jupyter notebook face_recognition_pipeline.ipynb
```

2. Pastikan variabel konfigurasi sesuai:

```python
CSV_FILE = 'dataset.csv'
DATA_DIR = 'Train'
BATCH_SIZE = 32
EPOCHS = 12
PATIENCE = 3
LEARNING_RATE = 0.001
IMG_SIZE = 480
```

3. Jalankan sel-sel secara berurutan untuk:  
   - Memuat CSV dan encode label.  
   - Membuat `Dataset` dan `DataLoader`.  
   - Membangun model EfficientNetV2-M + classifier baru.  
   - Training dengan early stopping dan tracking akurasi/kerugian.  
   - Evaluasi dengan confusion matrix dan classification report.[4]

## Catatan Teknis

- Notebook otomatis mendeteksi CUDA dan akan menggunakan GPU jika tersedia; jika tidak, training berjalan di CPU dengan peringatan dan saran reinstall PyTorch dengan CUDA support bila diperlukan.[4]
- Saat terjadi error membaca gambar, pipeline akan melaporkan path yang bermasalah dan mengganti dengan gambar hitam dummy untuk mencegah crash (bisa dimodifikasi sesuai kebutuhan).[4]
- Untuk deployment (misalnya presensi otomatis berbasis wajah), model dan mapping `idx_to_label` dapat disimpan dan digunakan di script inference terpisah.
