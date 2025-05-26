
# Image Classification with MobileNetV2

Proyek ini melakukan klasifikasi gambar menggunakan model MobileNetV2 dengan TensorFlow. Model dilatih menggunakan data gambar yang diproses dan diperkuat (augmentasi), kemudian disimpan dalam berbagai format: SavedModel, TFLite, dan TensorFlow.js (TFJS).

## 📂 Struktur Folder

```
├── saved_model/           # Model dalam format SavedModel
├── tflite/
│   ├── model.tflite       # Model TFLite
│   └── label.txt          # Label class
├── tfjs/
│   ├── model.json         # Model TFJS
│   └── group1-shard*      # Bobot model
├── requirements.txt       # Daftar dependensi
├── README.md              # Dokumentasi proyek ini
├── notebook.ipynb         # Notebook utama
```

## Menjalankan Proyek

### 1. Instalasi Dependensi

```bash
pip install -r requirements.txt
```

### 2. Jalankan Notebook

Buka `notebook.ipynb` dan jalankan semua sel. Notebook mencakup:
- Preprocessing dan augmentasi data
- Training awal dan fine-tuning
- Penyimpanan model ke SavedModel, TFLite, dan TFJS
- Inference dari gambar uji

## Inference Contoh

```python
img_path = "contoh.jpg"
# Lakukan prediksi dan tampilkan hasil
```

Contoh output:
```
Prediksi: Kucing (confidence: 0.95)
Label Asli: Kucing
```

## 📦 Format Model

| Format     | Platform                          |
|------------|-----------------------------------|
| SavedModel | TensorFlow Serving / Server       |
| TFLite     | Android / Embedded                |
| TFJS       | Web / JavaScript Application      |

---

## 👤 Author

- Nama: [MUHAMMAD AGUSRIANSYAH]
- Email: [misteragus46@gmail.com] / [mc224d5y1338@student.devacademy.id]
