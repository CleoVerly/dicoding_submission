# Tahap 0: Menginstal dan Mengimpor Library
# ==========================================
# Mengimpor semua library yang akan digunakan
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("Semua library berhasil diimpor!")


# Tahap 1: Memuat Dataset dan Memahami Data (Data Understanding)
# =============================================================
# Set path ke file CSV di dalam dataset Kaggle
file_path = "steam.csv"

# Memuat dataset menggunakan KaggleHub
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "nikdavis/steam-store-games",
    file_path
)

# Menampilkan ukuran awal dataset
print("Ukuran data awal:", df.shape)

# Menampilkan 5 baris pertama untuk melihat struktur data
print("\n5 data teratas:")
display(df.head())

# Memeriksa jumlah nilai yang hilang (missing values) di setiap kolom
print("\nJumlah missing value per kolom:")
print(df.isnull().sum())

# Memeriksa tipe data dari setiap kolom
print("\nTipe data masing-masing kolom:")
print(df.dtypes)


# Tahap 2: Persiapan Data (Data Preparation)
# ==========================================
# 1. Menghapus baris yang memiliki nilai kosong pada kolom-kolom penting
df.dropna(subset=['name', 'genres', 'steamspy_tags'], inplace=True)

# 2. Mengubah kolom 'release_date' menjadi format datetime
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

# 3. Menggabungkan fitur-fitur teks menjadi satu kolom 'content'
df['content'] = (
    df['genres'].fillna('') + ' ' +
    df['steamspy_tags'].fillna('') + ' ' +
    df['categories'].fillna('')
)

# 4. Mengambil sampel acak 5000 game untuk mengatasi keterbatasan RAM
df_final = df.sample(5000, random_state=42).reset_index(drop=True)

print(f"\nUkuran data setelah persiapan dan sampling: {df_final.shape}")


# Tahap 3: Modeling - Content-Based Filtering
# ============================================
# 1. TF-IDF Vectorization: Mengubah teks menjadi vektor numerik
# Membatasi jumlah fitur hingga 5000 untuk efisiensi memori
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(df_final['content'])

# 2. Cosine Similarity: Menghitung kemiripan antar game
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 3. Membuat "peta" dari nama game ke indeks barisnya untuk pencarian cepat
indices = pd.Series(df_final.index, index=df_final['name']).drop_duplicates()

# 4. Mendefinisikan fungsi untuk mendapatkan rekomendasi
def recommend(game_name, top_n=10):
    """Memberikan top-N rekomendasi game berdasarkan kemiripan konten."""
    # Cek apakah game input ada di dalam dataset sampel
    if game_name not in indices:
        return f"Game '{game_name}' tidak ditemukan di dalam data sampel."

    # Mendapatkan indeks dari game input
    idx = indices[game_name]

    # Mendapatkan skor kemiripan untuk semua game dengan game input
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Mengurutkan game berdasarkan skor kemiripan (dari tertinggi ke terendah)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Mengambil skor dari 10 game paling mirip (mengabaikan game itu sendiri di posisi 0)
    sim_scores = sim_scores[1:top_n+1]

    # Mendapatkan indeks dari game yang direkomendasikan
    game_indices = [i[0] for i in sim_scores]

    # Mengembalikan DataFrame dengan detail game yang direkomendasikan
    return df_final[['name', 'genres', 'steamspy_tags', 'price']].iloc[game_indices]

print("\nModel dan fungsi rekomendasi berhasil dibuat.")


# Tahap 4: Evaluasi Model
# =========================
def calculate_precision_at_k(input_game_name, recommendations, k=10):
    """Menghitung metrik Precision@K."""
    # Ambil genre dari game input
    input_game_genres = set(df_final[df_final['name'] == input_game_name]['genres'].iloc[0].split(';'))

    if not recommendations.empty:
        relevant_items = 0
        # Loop sebanyak k rekomendasi teratas
        for idx, row in recommendations.head(k).iterrows():
            recommended_genres = set(row['genres'].split(';'))
            # Cek apakah ada irisan genre (dianggap relevan)
            if input_game_genres.intersection(recommended_genres):
                relevant_items += 1
        
        return relevant_items / k
    return 0

# --- CONTOH PENGGUNAAN DAN EVALUASI ---
# 1. Pilih satu game secara acak dari dataset untuk diuji
test_game_name = df_final['name'].sample(1).iloc[0]
print(f"\n--- Memulai Evaluasi untuk Game: '{test_game_name}' ---")

# 2. Dapatkan rekomendasinya
top_10_recommendations = recommend(test_game_name, top_n=10)

# 3. Hitung presisinya
precision = calculate_precision_at_k(test_game_name, top_10_recommendations, k=10)

# 4. Tampilkan hasilnya
print("\nHasil rekomendasi:")
display(top_10_recommendations)
print(f"\nPrecision@10 untuk game '{test_game_name}' adalah: {precision:.0%}")

