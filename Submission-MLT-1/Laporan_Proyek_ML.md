# Laporan Proyek Machine Learning - Muhammad Agusriansyah

## Domain Proyek

Penyakit jantung merupakan salah satu penyebab utama kematian di seluruh dunia. Deteksi dini dan diagnosis yang akurat sangat penting untuk mengurangi risiko fatal dan meningkatkan kualitas hidup pasien. Proyek machine learning ini berfokus pada pemanfaatan data rekam medis untuk membangun model klasifikasi yang mampu memprediksi kemungkinan seseorang menderita penyakit jantung.

Menurut data dari WHO, penyakit kardiovaskular menyebabkan sekitar 17.9 juta kematian setiap tahunnya. Oleh karena itu, pengembangan sistem prediksi berbasis data diharapkan dapat menjadi alat bantu yang efektif bagi tenaga medis dalam proses skrining dan diagnosis awal.

Referensi:
[1] WHO. “Cardiovascular diseases (CVDs).” https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)

## Business Understanding

### Problem Statements
-   Bagaimana cara membangun model klasifikasi yang akurat untuk mendeteksi kemungkinan adanya penyakit jantung berdasarkan data atribut medis pasien?
-   Dari beberapa algoritma klasifikasi yang diuji, algoritma manakah yang memberikan performa terbaik dalam memprediksi penyakit jantung pada dataset ini?

### Goals
-   Membangun sebuah model machine learning yang dapat memprediksi keberadaan penyakit jantung dengan tingkat akurasi yang tinggi.
-   Membandingkan dan mengevaluasi performa dari tiga algoritma klasifikasi yang berbeda: Random Forest, Decision Tree, dan K-Nearest Neighbors (KNN).
-   Mengidentifikasi model terbaik setelah dilakukan optimasi hyperparameter.

### Solution Statements
-   Melakukan pra-pemrosesan data untuk menyiapkan dataset agar siap digunakan untuk pemodelan.
-   Melatih tiga model klasifikasi (Random Forest, Decision Tree, KNN) pada data medis pasien.
-   Meningkatkan performa setiap model melalui proses *hyperparameter tuning* menggunakan `GridSearchCV`.
-   Mengevaluasi dan membandingkan model berdasarkan metrik evaluasi seperti akurasi, presisi, recall, dan F1-score untuk menentukan solusi terbaik.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah "Heart Failure Prediction" yang bersumber dari platform Kaggle. Dataset ini berisi **918 sampel data (baris) dan 12 kolom**. [cite_start]Dataset dapat diakses publik melalui URL berikut: [https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)[cite: 10, 14].

[cite_start]Berikut adalah deskripsi untuk setiap fitur yang ada dalam dataset awal:
* **Age**: Umur pasien (tahun).
* **Sex**: Jenis kelamin pasien (M: Laki-laki, F: Perempuan).
* **ChestPainType**: Tipe nyeri dada yang dialami (TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic).
* **RestingBP**: Tekanan darah istirahat pasien (mm Hg).
* **Cholesterol**: Kadar kolesterol serum (mm/dl).
* [cite_start]**FastingBS**: Gula darah puasa (1: jika FastingBS > 120 mg/dl, 0: sebaliknya).
* [cite_start]**RestingECG**: Hasil rekaman elektrokardiogram (EKG) saat istirahat (Normal: Normal, ST: memiliki kelainan gelombang ST-T, LVH: menunjukkan kemungkinan hipertrofi ventrikel kiri).
* [cite_start]**MaxHR**: Detak jantung maksimum yang tercapai (nilai numerik).
* [cite_start]**ExerciseAngina**: Terjadinya angina (nyeri dada) akibat olahraga (Y: Ya, N: Tidak).
* [cite_start]**Oldpeak**: Depresi ST yang diinduksi oleh olahraga relatif terhadap istirahat (nilai numerik).
* [cite_start]**ST_Slope**: Kemiringan segmen ST pada puncak olahraga (Up: menanjak, Flat: datar, Down: menurun).
* **HeartDisease**: Variabel target yang menunjukkan keberadaan penyakit jantung (1: Ya, 0: Tidak).

Berdasarkan eksplorasi data awal pada notebook, distribusi kelas target cukup seimbang dan tidak ditemukan adanya nilai yang hilang (*missing values*).

## Data Preparation

Tahapan persiapan data dilakukan untuk mengubah data mentah menjadi format yang siap digunakan oleh model machine learning. Berikut adalah langkah-langkah yang dilakukan secara berurutan sesuai dengan implementasi pada notebook:

1.  **Encoding Fitur Kategorikal**: Melakukan *one-hot encoding* pada kolom-kolom dengan tipe data object menggunakan fungsi `pd.get_dummies()`.
2.  **Encoding Variabel Target**: Mengubah label pada kolom target `HeartDisease` menjadi format numerik (0 dan 1) menggunakan `LabelEncoder`.
3.  **Standardisasi Fitur Numerik**: Menyamakan skala semua fitur numerik menggunakan `StandardScaler`. Proses ini penting agar fitur dengan rentang nilai yang besar tidak mendominasi proses pelatihan model.
4.  **Pembagian Data**: Membagi data menjadi data latih (80%) dan data uji (20%) menggunakan `train_test_split` untuk mengevaluasi performa model pada data yang belum pernah dilihat.

## Modeling

Tiga algoritma klasifikasi digunakan dalam proyek ini. Berikut adalah penjelasan singkat mengenai cara kerja setiap algoritma yang digunakan:
-   **Random Forest**: Algoritma ini bekerja dengan membangun sejumlah besar Decision Tree (pohon keputusan) pada berbagai sub-sampel dari dataset. Untuk prediksi klasifikasi, setiap pohon memberikan suara (voting), dan kelas dengan suara terbanyak menjadi prediksi akhir dari model. Hal ini membantu mengurangi overfitting yang sering terjadi pada satu Decision Tree.
-   **Decision Tree**: Algoritma ini bekerja dengan memecah dataset menjadi himpunan bagian yang lebih kecil secara berulang berdasarkan fitur-fitur yang ada, hingga mencapai keputusan di daun pohon (leaf). Setiap pemecahan didasarkan pada kriteria tertentu (misalnya Gini impurity atau information gain) untuk menghasilkan kelompok yang paling homogen.
-   **K-Nearest Neighbors (KNN)**: Algoritma ini bekerja dengan mengklasifikasikan data baru berdasarkan mayoritas kelas dari 'K' tetangga terdekatnya dalam ruang fitur. Jarak antar data dihitung menggunakan metrik tertentu (misalnya jarak Euclidean). KNN adalah algoritma *lazy learning* karena tidak membangun model secara eksplisit, melainkan hanya menyimpan data latih untuk melakukan prediksi.

Proses pemodelan dilakukan dalam dua tahap:
1.  **Training Awal**: Setiap model dilatih menggunakan parameter default dari library Scikit-learn untuk mendapatkan performa dasar (baseline).
    * Random Forest: `random_state=42`
    * Decision Tree: `random_state=42`
    * K-Nearest Neighbors: (menggunakan parameter default)
2.  **Hyperparameter Tuning**: Performa setiap model ditingkatkan dengan mencari kombinasi hyperparameter terbaik menggunakan `GridSearchCV` dengan 5-fold cross-validation (`cv=5`). Parameter yang diuji untuk setiap model adalah sebagai berikut:
    * **Random Forest**:
        * `n_estimators`: `[50, 100, 150]`
        * `max_depth`: `[None, 5, 10, 20]`
        * `min_samples_split`: `[2, 5, 10]`
        * `min_samples_leaf`: `[1, 2, 4]`
    * **Decision Tree**:
        * `criterion`: `['gini', 'entropy']`
        * `max_depth`: `[None, 5, 10, 20]`
        * `min_samples_split`: `[2, 5, 10]`
        * `min_samples_leaf`: `[1, 2, 4]`
    * **K-Nearest Neighbors**:
        * `n_neighbors`: `[3, 5, 7, 9]`
        * `weights`: `['uniform', 'distance']`
        * `p`: `[1, 2]` (metrik jarak)

## Evaluation

[cite_start]Model dievaluasi berdasarkan metrik Accuracy, Precision, Recall, dan F1-score pada data uji. Tabel berikut merangkum hasil akurasi dari setiap model setelah proses hyperparameter tuning, sesuai dengan hasil yang didapatkan pada notebook.

| Model                 | Akurasi (setelah tuning) |
| --------------------- | ------------------------ |
| Random Forest         | 0.8641                   |
| Decision Tree         | 0.8207                   |
| K-Nearest Neighbors   | **0.8859** |

**Model Terbaik:**
[cite_start]Berdasarkan hasil akurasi pada data uji, **K-Nearest Neighbors (KNN)** yang telah di-tuning adalah model terbaik dengan akurasi **0.8859**. [cite_start]Berikut adalah rincian metrik evaluasi untuk model KNN terbaik:

| Kelas               | Precision | Recall | F1-Score |
| ------------------- | --------- | ------ | -------- |
| Tidak Sakit Jantung | 0.85      | 0.88   | 0.87     |
| Sakit Jantung       | 0.91      | 0.89   | 0.90     |

Secara keseluruhan, model KNN yang telah dioptimalkan tidak hanya memberikan akurasi tertinggi tetapi juga menunjukkan F1-score yang kuat dan seimbang untuk kedua kelas, menjadikannya solusi terbaik untuk masalah prediksi penyakit jantung ini.

---
Proyek ini menunjukkan bahwa model machine learning, khususnya K-Nearest Neighbors setelah dioptimalkan, mampu memprediksi risiko penyakit jantung dengan akurasi yang tinggi, yang berpotensi mendukung praktik diagnosa lebih dini di bidang medis.