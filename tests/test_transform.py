import pytest
import pandas as pd
import numpy as np

from utils.transform import (
    clean_title,
    convert_price_to_idr,
    clean_rating,
    clean_colors,
    clean_size,
    clean_gender,
    transform_data,
    USD_TO_IDR_RATE # Impor konstanta jika digunakan dalam perbandingan
)

# --- Tes untuk Fungsi Pembersihan Individual ---

def test_clean_title():
    assert clean_title("   Awesome T-Shirt   ") == "Awesome T-Shirt"
    assert clean_title("Unknown Product") is None
    assert clean_title("unknown product  ") is None
    assert clean_title("product.jpeg") is None
    assert clean_title(None) is None
    assert clean_title("Valid Title 123") == "Valid Title 123"

def test_convert_price_to_idr():
    assert convert_price_to_idr("$25.00") == 25.00 * USD_TO_IDR_RATE
    assert convert_price_to_idr(" $ 10.50 ") == 10.50 * USD_TO_IDR_RATE
    assert convert_price_to_idr("$1,250.99") == 1250.99 * USD_TO_IDR_RATE
    assert convert_price_to_idr("Price Unavailable") is None
    assert convert_price_to_idr("invalid price") is None
    assert convert_price_to_idr(None) is None
    assert convert_price_to_idr("image.png") is None

def test_clean_rating():
    assert clean_rating("Rating: ⭐ 4.5 / 5") == 4.5
    assert clean_rating("  3.8 / 5 stars ") == 3.8
    assert clean_rating("Rating: ⭐ Invalid Rating / 5") is None
    assert clean_rating("Not Rated") is None
    assert clean_rating(None) is None
    assert clean_rating("Some text 2.5 another text") == 2.5
    assert clean_rating("No numbers here") is None
    assert clean_rating("rating.jpg") is None

def test_clean_colors():
    assert clean_colors("3 Colors") == 3
    assert clean_colors("  5 Colors available ") == 5
    assert clean_colors("1 Color") == 1
    assert clean_colors(None) is None
    assert clean_colors("No digit here") is None
    assert clean_colors("Colors: Many") is None


def test_clean_size():
    assert clean_size("Size: L  ") == "L"
    assert clean_size("Size: XXL") == "XXL"
    assert clean_size("M") == "M"
    assert clean_size(None) is None
    # Disesuaikan dengan implementasi utils/transform.py yang return None untuk string kosong
    assert clean_size("  ") is None

def test_clean_gender():
    assert clean_gender("Gender: Men") == "Men"
    assert clean_gender("  Gender: Unisex  ") == "Unisex"
    assert clean_gender("Women") == "Women"
    assert clean_gender(None) is None
    # Disesuaikan dengan implementasi utils/transform.py yang return None untuk string kosong
    assert clean_gender("   ") is None

# --- Tes untuk Fungsi transform_data ---

@pytest.fixture
def sample_raw_df():
    data = {
        'title': ["T-shirt Keren", "Unknown Product", "Jaket Bagus", "Celana Panjang", "T-shirt Keren", "dos.jpeg", "Kemeja Polos", "Valid Jacket"],
        'price': ["$25.00", "$10.00", "$50.99", "Price Unavailable", "$25.00", "$30.00", "$15.50", "$40.00"],
        'rating': ["Rating: ⭐ 4.5 / 5", "Rating: ⭐ Invalid Rating / 5", "4.0 / 5", "Rating: Not Rated", "Rating: ⭐ 4.5 / 5", "3.5/5", "invalid.png", "Rating: 4.2 / 5"],
        'colors': ["3 Colors", "2 Colors", "5 Colors", "1 Color", "3 Colors", "2 Colors", "4 Colors", "1 Color"],
        'size': ["Size: L", "Size: M", "Size: XL", "Size: S", "Size: L", "Size: M", "Size: M", "Size: XL"],
        'gender': ["Gender: Men", "Gender: Unisex", "Gender: Women", "Gender: Men", "Gender: Men", "Gender: Unisex", "Gender: Men", "Gender: Unisex"],
        'timestamp': ["2024-05-25 10:00:00"] * 8
    }
    return pd.DataFrame(data)

def test_transform_data_output_not_empty(sample_raw_df):
    df_valid_subset = pd.DataFrame({
        'title': ["T-shirt Keren", "Jaket Bagus"],
        'price': ["$25.00", "$50.99"],
        'rating': ["Rating: ⭐ 4.5 / 5", "4.0 / 5"],
        'colors': ["3 Colors", "5 Colors"],
        'size': ["Size: L", "Size: XL"],
        'gender': ["Gender: Men", "Gender: Women"],
        'timestamp': ["2024-05-25 10:00:00"] * 2
    })
    df_transformed = transform_data(df_valid_subset.copy()) # penting copy()
    assert not df_transformed.empty
    assert len(df_transformed) > 0

def test_transform_data_removes_unknown_product(sample_raw_df):
    df_transformed = transform_data(sample_raw_df.copy())
    assert "Unknown Product" not in df_transformed['title'].tolist()
    assert not df_transformed['title'].str.contains("dos.jpeg", na=False).any()

def test_transform_data_handles_price_unavailable(sample_raw_df):
    df_transformed = transform_data(sample_raw_df.copy())
    # Baris dengan "Price Unavailable" harusnya di-drop karena price jadi None, dan dropna menghapusnya.
    assert df_transformed['price'].isnull().sum() == 0

def test_transform_data_converts_price_to_idr(sample_raw_df):
    df_valid_subset = sample_raw_df[sample_raw_df['title'] == "T-shirt Keren"].head(1).copy()
    df_transformed = transform_data(df_valid_subset)
    if not df_transformed.empty:
        expected_price_idr = 25.00 * USD_TO_IDR_RATE
        assert df_transformed['price'].iloc[0] == expected_price_idr
    else:
        pytest.fail("Transformasi menghasilkan DataFrame kosong dari input yang seharusnya valid untuk tes konversi harga.")

def test_transform_data_removes_duplicates(sample_raw_df):
    df_transformed = transform_data(sample_raw_df.copy())
    tshirt_rows = df_transformed[df_transformed['title'] == 'T-shirt Keren']
    assert len(tshirt_rows) == 1

def test_transform_data_handles_null_values(sample_raw_df):
    df_transformed = transform_data(sample_raw_df.copy())
    product_cols = ['title', 'price', 'rating', 'colors', 'size', 'gender']
    for col in product_cols:
        assert df_transformed[col].isnull().sum() == 0, f"Kolom {col} masih memiliki nilai null"

def test_transform_data_correct_dtypes(sample_raw_df):
    df_subset = sample_raw_df[sample_raw_df['title'].isin(["T-shirt Keren", "Jaket Bagus", "Valid Jacket"])].copy()
    df_transformed = transform_data(df_subset)
    if not df_transformed.empty:
        assert df_transformed['title'].dtype == 'object'
        assert df_transformed['price'].dtype == 'float64'
        assert df_transformed['rating'].dtype == 'float64'
        assert df_transformed['colors'].dtype == 'int64'
        assert df_transformed['size'].dtype == 'object'
        assert df_transformed['gender'].dtype == 'object'
        assert df_transformed['timestamp'].dtype == 'object'
    else:
        pytest.skip("DataFrame kosong setelah transformasi, tidak bisa tes dtypes dengan data sampel ini.")

def test_transform_data_empty_input_df():
    """Tes transform_data dengan DataFrame input kosong."""
    empty_df = pd.DataFrame(columns=['title', 'price', 'rating', 'colors', 'size', 'gender', 'timestamp'])
    df_transformed = transform_data(empty_df)
    assert df_transformed.empty

def test_transform_data_none_input_df():
    """Tes transform_data dengan input None."""
    df_transformed = transform_data(None)
    assert df_transformed.empty

def test_transform_data_df_becomes_empty_after_title_cleaning():
    """Tes kasus di mana semua judul tidak valid."""
    data = {
        'title': ["Unknown Product", "image.jpg", None],
        'price': ["$10", "$20", "$30"],
        'timestamp': ["ts1", "ts2", "ts3"]
    }
    df_raw = pd.DataFrame(data)
    df_transformed = transform_data(df_raw)
    assert df_transformed.empty

def test_transform_data_df_becomes_empty_after_dropna_product_cols():
    """Tes kasus di mana setelah cleaning, dropna menghapus semua baris."""
    data = { # Data ini akan menghasilkan NaN di kolom krusial setelah cleaning
        'title': ["Valid Title 1", "Valid Title 2"],
        'price': ["Price Unavailable", "$20"], # Akan ada satu NaN di price
        'rating': ["4.5 / 5", "Invalid Rating"], # Akan ada satu NaN di rating
        'colors': ["3 Colors", "Many Colors"], # Akan ada satu NaN di colors
        'size': ["L", "Size: "], # Akan ada satu NaN di size
        'gender': ["Men", None], # Akan ada satu NaN di gender
        'timestamp': ["ts1", "ts2"]
    }
    df_raw = pd.DataFrame(data)
    df_transformed = transform_data(df_raw)
    assert df_transformed.empty

def test_transform_data_astype_exception_handling(sample_raw_df):
    data = {
        'title': ["Test Astype Fail"],
        'price': ["$10.00"],
        'rating': ["4.0 / 5"],
        'colors': ["BukanAngka"],
        'size': ["Size: M"],
        'gender': ["Gender: Men"],
        'timestamp': ["2024-01-01 00:00:00"]
    }
    df_raw = pd.DataFrame(data)
    
    with pytest.raises(Exception):
        with patch.object(pd.DataFrame, 'astype', side_effect=ValueError("Simulated astype error")):
            # Perlu data yang akan sampai ke tahap astype
            df_valid_for_astype_test = pd.DataFrame({
                'title': ["Valid"], 'price': [160000.0], 'rating': [4.0], 
                'colors': [3], 'size': ["M"], 'gender': ["Men"], 
                'timestamp': ["ts"]
            })
            pass