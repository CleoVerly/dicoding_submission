import pandas as pd
import numpy as np
import re

# Asumsi nilai tukar
USD_TO_IDR_RATE = 16000

def clean_title(title):
    """Membersihkan judul."""
    if pd.isna(title):
        return None
    title_stripped = str(title).strip()
    # Menghapus produk dengan judul "Unknown Product"
    if title_stripped.lower() == "unknown product":
        return None
    # Cek apakah title mengandung ekstensi file gambar
    if re.search(r'\.(jpeg|jpg|png|gif)$', title_stripped.lower()):
        return None
    return title_stripped

def convert_price_to_idr(price_str):

    if pd.isna(price_str) or str(price_str).strip().lower() == "price unavailable":
        return None
    
    # Menghilangkan simbol '$' dan koma
    cleaned_price = str(price_str).replace('$', '').replace(',', '').strip()
    try:
        price_usd = float(cleaned_price)
        return price_usd * USD_TO_IDR_RATE
    except ValueError:

        if re.search(r'\.(jpeg|jpg|png|gif)$', cleaned_price.lower()):
            return None
        return None

def clean_rating(rating_str):
    
    if pd.isna(rating_str):
        return None
    
    rating_text = str(rating_str).strip()
    # Menangani "Invalid Rating" atau "Not Rated"
    if "invalid rating" in rating_text.lower() or "not rated" in rating_text.lower():
        return None
    
    match = re.search(r'(\d+(\.\d+)?)', rating_text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None

    if re.search(r'\.(jpeg|jpg|png|gif)$', rating_text.lower()):
        return None
    return None

def clean_colors(colors_str):
    
    if pd.isna(colors_str):
        return None
    match = re.search(r'\d+', str(colors_str))
    if match:
        try:
            return int(match.group(0))
        except ValueError:
            return None
    return None

def clean_size(size_str):
    
    if pd.isna(size_str):
        return None
    cleaned_size = re.sub(r'Size:\s*', '', str(size_str), flags=re.IGNORECASE).strip()
    return cleaned_size if cleaned_size else None

def clean_gender(gender_str):
    
    if pd.isna(gender_str):
        return None
    cleaned_gender = re.sub(r'Gender:\s*', '', str(gender_str), flags=re.IGNORECASE).strip()
    return cleaned_gender if cleaned_gender else None

def transform_data(df_raw):
    if df_raw is None or df_raw.empty:
        print("DataFrame mentah kosong atau None, tidak ada data untuk ditransformasi.")
        return pd.DataFrame()

    print(f"Memulai transformasi data. Jumlah data mentah: {len(df_raw)}")
    df = df_raw.copy()

    expected_cols_from_extract = ['title', 'price', 'rating', 'colors', 'size', 'gender', 'timestamp']
    for col in expected_cols_from_extract:
        if col not in df.columns:
            df[col] = None

    # 1. Bersihkan Title
    df['title'] = df['title'].apply(clean_title)
    # Hapus baris jika judul menjadi None setelah dibersihkan
    df.dropna(subset=['title'], inplace=True)
    print(f"Data setelah membersihkan title 'Unknown Product' dan invalid: {len(df)}")
    if df.empty: return pd.DataFrame() # Jika semua data invalid

    # 2. Konversi Harga ke IDR
    df['price'] = df['price'].apply(convert_price_to_idr)

    # 3. Bersihkan kolom Rating, Colors, Size, Gender
    df['rating'] = df['rating'].apply(clean_rating)
    df['colors'] = df['colors'].apply(clean_colors)
    df['size'] = df['size'].apply(clean_size)
    df['gender'] = df['gender'].apply(clean_gender)
    
    # 4. Penanganan Duplikat
    product_cols = ['title', 'price', 'rating', 'colors', 'size', 'gender']
    df.drop_duplicates(subset=product_cols, keep='first', inplace=True)
    print(f"Data setelah menghapus duplikat: {len(df)}")
    if df.empty: return pd.DataFrame()

    # 5. Penanganan Nilai Hilang (Null/NaN)
    df.dropna(subset=product_cols, inplace=True)
    print(f"Data setelah menghapus baris dengan nilai null pada kolom produk inti: {len(df)}")
    if df.empty: return pd.DataFrame()
    
    # 6. Konversi Tipe Data Final
    try:
        df = df.astype({
            'title': 'object',
            'price': 'float64',
            'rating': 'float64',
            'colors': 'int64',
            'size': 'object',
            'gender': 'object',
            'timestamp': 'object'
        })
    except Exception as e:
        print(f"Error saat konversi tipe data final: {e}")

    # Reset index setelah dropna dan drop_duplicates
    df.reset_index(drop=True, inplace=True)

    print(f"Transformasi selesai. Jumlah data bersih: {len(df)}")
    return df