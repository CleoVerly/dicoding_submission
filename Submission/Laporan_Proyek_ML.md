
# Laporan Proyek Machine Learning - Muhammad Agusriansyah

## Domain Proyek

Penyakit jantung merupakan penyebab utama kematian di seluruh dunia. Deteksi dini dan diagnosis berbasis data medis sangat penting untuk mencegah risiko fatal. Proyek ini berfokus pada pemanfaatan data rekam medis pasien untuk memprediksi kemungkinan penyakit jantung dengan menggunakan algoritma klasifikasi Machine Learning.

Menurut WHO, sekitar 17.9 juta orang meninggal setiap tahun akibat penyakit kardiovaskular [1]. Oleh karena itu, sistem prediksi berbasis data sangat dibutuhkan untuk membantu proses diagnosa oleh tenaga medis.

Referensi:  
[1] WHO. “Cardiovascular diseases (CVDs).” https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)

## Business Understanding

### Problem Statements
- Bagaimana membangun model klasifikasi untuk mendeteksi kemungkinan penyakit jantung?
- Algoritma manakah yang memberikan performa terbaik untuk klasifikasi ini?

### Goals
- Membangun sistem prediksi penyakit jantung menggunakan data medis.
- Membandingkan performa tiga model: Random Forest, Decision Tree, dan K-Nearest Neighbors.

### Solution Statements
- Melatih dan mengevaluasi tiga model klasifikasi (RF, DT, KNN).
- Meningkatkan akurasi melalui hyperparameter tuning menggunakan GridSearchCV.
- Mengukur performa berdasarkan metrik akurasi, precision, recall, dan F1-score.

## Data Understanding

Dataset yang digunakan berasal dari Kaggle dan berisi 918 sampel data pasien. Dataset ini memiliki fitur-fitur seperti umur, jenis kelamin, tekanan darah, kolesterol, dll., serta target berupa `HeartDisease` (0 = tidak, 1 = ya).

Contoh fitur:
- Age: Umur pasien
- Sex: Jenis kelamin
- ChestPainType: Jenis nyeri dada
- RestingBP: Tekanan darah istirahat
- Cholesterol: Tingkat kolesterol
- HeartDisease: Target (0 atau 1)

Distribusi kelas target cukup seimbang dan tidak mengandung missing value.

## Data Preparation

- Melakukan **encoding** pada kolom kategorikal menggunakan `pd.get_dummies()`.
- Melakukan **standardisasi** fitur numerik menggunakan `StandardScaler`.
- Melakukan encoding pada target `HeartDisease` menggunakan `LabelEncoder`.
- Membagi data menjadi training dan testing set (80:20 stratified).

## Modeling

Tiga algoritma yang digunakan:
- Random Forest
- Decision Tree
- K-Nearest Neighbors

Pemodelan dilakukan dua tahap:
1. **Training awal** tanpa tuning.
2. **Hyperparameter tuning** dengan `GridSearchCV` (CV=3).

Setiap model dievaluasi menggunakan confusion matrix dan classification report. Hasil evaluasi disimpan dalam bentuk `.png`.

## Evaluation

Metrik evaluasi yang digunakan:
- Accuracy
- Precision
- Recall
- F1-score

Contoh hasil akurasi model (tuned):
- Random Forest: 0.8859
- Decision Tree: 0.8587
- KNN: 0.8370

Model terbaik adalah **Random Forest** berdasarkan akurasi dan skor evaluasi lain.

---

_Proyek ini menunjukkan bahwa model Machine Learning mampu memprediksi risiko penyakit jantung dengan akurasi tinggi, mendukung praktik diagnosa lebih dini di bidang medis._
