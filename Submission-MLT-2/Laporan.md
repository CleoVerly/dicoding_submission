# Laporan Proyek Machine Learning - Sistem Rekomendasi Game

**Nama:** MUHAMMAD AGUSRIANSYAH
**Email:** mc224d5y1338@student.devacademy.id
**ID Dicoding:** mc224d5y1338

---

## Project Overview

Industri game digital, khususnya melalui platform seperti Steam, telah mengalami pertumbuhan eksponensial. Steam menampung puluhan ribu game yang tersedia untuk dibeli dan dimainkan, menciptakan tantangan besar bagi pengguna untuk menemukan game baru yang sesuai dengan selera mereka. Fenomena ini dikenal sebagai *information overload*, di mana pengguna kesulitan membuat keputusan karena terlalu banyaknya pilihan yang tersedia.

Proyek ini bertujuan untuk mengatasi masalah tersebut dengan membangun sebuah sistem rekomendasi. Sistem ini akan membantu pengguna menemukan game yang mungkin mereka sukai berdasarkan kemiripan konten dengan game yang pernah mereka mainkan atau ketahui. Dengan adanya sistem ini, diharapkan dapat meningkatkan pengalaman pengguna dalam menavigasi perpustakaan game yang sangat besar dan membantu mereka menemukan "permata tersembunyi" yang mungkin terlewatkan.

---

## Business Understanding

### Problem Statements

* Bagaimana cara membantu pengguna mengatasi *information overload* di platform dengan puluhan ribu game?
* Bagaimana cara memberikan rekomendasi game yang relevan dan personal kepada pengguna berdasarkan game yang mereka sukai?
* Bagaimana cara mengidentifikasi game-game yang memiliki kemiripan konten (seperti genre atau fitur permainan) satu sama lain secara efisien?

### Goals

* Membangun sebuah model sistem rekomendasi yang dapat menghasilkan daftar game yang relevan berdasarkan input satu game.
* Menggunakan pendekatan *Content-Based Filtering* untuk merekomendasikan game berdasarkan fitur-fitur intrinsik seperti genre, tag, dan kategori.
* Menghasilkan prototipe sistem rekomendasi yang dapat dievaluasi kinerjanya menggunakan metrik yang relevan.

### Solution Statements

1. **Content-Based Filtering (Pendekatan yang Dipilih)**
   Pendekatan ini akan merekomendasikan item (game) berdasarkan kemiripan atribut atau fitur dari item itu sendiri. Fitur yang digunakan adalah `genres`, `categories`, dan `steamspy_tags`. Pendekatan ini dipilih karena dataset kaya akan fitur konten dan tidak memerlukan data interaksi pengguna (seperti rating). Prosesnya melibatkan teknik TF-IDF untuk mengubah data teks menjadi vektor dan Cosine Similarity untuk mengukur kemiripan antar vektor tersebut.

2. **Collaborative Filtering**
   Pendekatan ini merekomendasikan item berdasarkan pola perilaku pengguna di masa lalu. Namun, karena dataset ini tidak memiliki interaksi pengguna secara eksplisit, pendekatan ini tidak menjadi fokus utama.

---

## Data Understanding

Dataset yang digunakan adalah [Steam Store Games - Kaggle](https://www.kaggle.com/datasets/nikdavis/steam-store-games) yang memuat lebih dari 27.000 game dari platform Steam. Dataset memiliki **27.075 baris** dan **18 kolom**.

### Variabel-variabel pada Data

* `appid` : ID unik untuk masing-masing game
* `name` : Nama game
* `release_date` : Tanggal rilis
* `english` : Dukungan bahasa Inggris (1: ya, 0: tidak)
* `developer` : Pengembang
* `publisher` : Penerbit
* `platforms` : Platform yang didukung
* `required_age` : Usia minimum
* `categories` : Fitur game (Single-player, dll.)
* `genres` : Genre utama
* `steamspy_tags` : Tag tambahan
* `achievements` : Jumlah pencapaian
* `positive_ratings` : Ulasan positif
* `negative_ratings` : Ulasan negatif
* `average_playtime` : Rata-rata waktu bermain
* `median_playtime` : Median waktu bermain
* `owners` : Estimasi jumlah pemilik
* `price` : Harga game (USD)

### Exploratory Data Analysis (EDA)

* Ditemukan nilai kosong pada kolom `developer` dan `publisher`
* Kolom `release_date` awalnya bertipe `object`, perlu dikonversi ke `datetime`

```python
# Cek missing values
df.isnull().sum()

# Cek tipe data
df.dtypes
```

---

## Data Preparation

## Data Preparation

Langkah-langkah persiapan data yang dilakukan bertujuan untuk membersihkan, mentransformasi, dan menyiapkan data agar siap digunakan untuk pemodelan. Berikut adalah rincian tahapannya:

* **Menghapus Baris Kosong**: Menghapus baris dengan nilai kosong pada kolom-kolom yang esensial untuk rekomendasi, yaitu `name`, `genres`, dan `steamspy_tags`, untuk memastikan kualitas fitur yang akan digunakan.
* **Konversi Tipe Data**: Mengubah kolom `release_date` dari tipe `object` menjadi format `datetime` untuk analisis berbasis waktu jika diperlukan di masa depan.
* **Penggabungan Fitur (Feature Engineering)**: Membuat satu fitur utama bernama `content` dengan menggabungkan teks dari kolom `genres`, `steamspy_tags`, dan `categories`.
* **Penanganan Nilai Kosong**: Sebelum digabungkan, nilai kosong (NaN) pada kolom `genres`, `steamspy_tags`, dan `categories` diisi dengan string kosong (`''`). Langkah ini penting untuk memastikan proses penggabungan string berjalan lancar dan tidak ada data yang hilang karena nilai null.
* **Vektorisasi TF-IDF**: Mengubah data teks pada kolom `content` menjadi representasi numerik dalam bentuk matriks vektor menggunakan `TfidfVectorizer`. TF-IDF (Term Frequency-Inverse Document Frequency) adalah teknik yang memberikan bobot pada setiap kata berdasarkan frekuensinya dalam satu dokumen (game) dan kelangkaannya di seluruh dokumen (semua game). Proses ini merupakan langkah akhir dalam persiapan data sebelum data dimasukkan ke model.
* **Sampling Data**: Mengambil sampel sebanyak 5.000 data secara acak. Langkah ini dilakukan untuk mengelola penggunaan memori (RAM) dan memastikan proses pemodelan berjalan lebih efisien tanpa mengorbankan representasi data secara signifikan.

Berikut adalah kode yang mencerminkan langkah-langkah di atas:

```python
# Hapus baris dengan nilai penting yang kosong
df = df.dropna(subset=['name', 'genres', 'steamspy_tags'])

# Konversi tipe data tanggal
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

# Gabungkan fitur konten dengan penanganan nilai kosong
df['content'] = (
    df['genres'].fillna('') + ' ' +
    df['steamspy_tags'].fillna('') + ' ' +
    df['categories'].fillna('')
)

# Ambil sampel 5000 data untuk efisiensi
df_final = df.sample(5000, random_state=42).reset_index(drop=True)

# Lakukan TF-IDF Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(df_final['content'])

---

## Modeling

Model sistem rekomendasi dibangun menggunakan pendekatan **Content-Based Filtering**. Setelah data disiapkan dan diubah menjadi matriks TF-IDF pada tahap Data Preparation, bagian Modeling berfokus pada bagaimana rekomendasi dihasilkan dari data tersebut.

Langkah-langkah pemodelan adalah sebagai berikut:

* **Menghitung Kemiripan Konten**: Menggunakan metrik **Cosine Similarity** untuk menghitung derajat kesamaan antara setiap pasang game. Cosine Similarity mengukur kosinus sudut antara dua vektor (dalam hal ini, vektor TF-IDF dari setiap game), yang menghasilkan skor kemiripan antara 0 (tidak mirip) dan 1 (sangat mirip).
* **Membuat Fungsi Rekomendasi**: Membangun fungsi `recommend()` yang menerima input nama game dan mengembalikan daftar *top-N* game lain yang paling mirip. Fungsi ini bekerja dengan cara:
    1.  Mencari indeks dari game yang diinput.
    2.  Mengambil skor kemiripan game tersebut dengan semua game lainnya dari matriks Cosine Similarity.
    3.  Mengurutkan game berdasarkan skor kemiripan tertinggi.
    4.  Menampilkan N game teratas sebagai hasil rekomendasi.

### Contoh Hasil Rekomendasi

Input: *Dragon's Dogma: Dark Arisen*
Rekomendasi 10 teratas:

| No | Nama Game                 | Genre                  | Harga (USD) |
| -- | ------------------------- | ---------------------- | ----------- |
| 1  | Way of the Samurai 3      | Action;Adventure       | 14.99       |
| 2  | Skyrim Special Edition    | RPG                    | 29.99       |
| 3  | ELEX                      | Action;Adventure;RPG   | 39.99       |
| 4  | Shadow of War             | Action;Adventure;RPG   | 34.99       |
| 5  | Kingdom Come: Deliverance | Action;Adventure;RPG   | 29.99       |
| 6  | Sunset Overdrive          | Action;Adventure       | 14.99       |
| 7  | Oceanhorn                 | Action;Adventure;Indie | 10.99       |
| 8  | Mount & Blade             | Action;RPG             | 7.99        |
| 9  | Drifting Lands            | Action;Indie;RPG       | 13.99       |
| 10 | Phoning Home              | Action;Adventure;Indie | 14.99       |

### Kelebihan:

* Tidak memerlukan data pengguna
* Cocok untuk item baru
* Rekomendasi mudah dijelaskan

### Kekurangan:

* Kurang variasi (serendipity rendah)
* Terbatas pada fitur konten

---

## Evaluation

### Metrik Evaluasi: Precision\@K

Precision\@K mengukur proporsi item relevan dari K rekomendasi teratas. Game dianggap relevan jika memiliki setidaknya satu genre yang sama.

**Formula:**

$\text{Precision@K} = \frac{\text{Jumlah item relevan dalam top-K}}{K}$

### Hasil Evaluasi

Pada pengujian dengan game *Dragon's Dogma: Dark Arisen*, model menghasilkan **Precision\@10 = 100%**.

Artinya, semua game dalam daftar top-10 memiliki setidaknya satu genre yang sama, menunjukkan efektivitas model dalam menemukan game yang mirip secara konten.

---

**MUHAMMAD AGUSRIANSYAH**
