import pytest
import pandas as pd
import os

from utils.load import load_to_csv

@pytest.fixture
def sample_transformed_df():
    """Fixture untuk DataFrame sampel yang sudah bersih."""
    data = {
        'title': ["T-shirt Keren", "Jaket Bagus", "Celana Panjang"],
        'price': [400000.0, 815840.0, 250000.0],
        'rating': [4.5, 4.0, 3.0],
        'colors': [3, 5, 1],
        'size': ["L", "XL", "S"],
        'gender': ["Men", "Women", "Men"],
        'timestamp': ["2024-05-25 10:00:00", "2024-05-25 10:00:00", "2024-05-25 10:00:00"]
    }
    return pd.DataFrame(data)

def test_load_to_csv_successful_creation(sample_transformed_df, tmp_path):
    """
    Tes apakah load_to_csv berhasil membuat file CSV dengan konten yang benar.
    tmp_path adalah fixture pytest untuk path temporer.
    """
    output_filename = "test_products.csv"
    # Menggunakan tmp_path (objek Pathlib) untuk membuat path file
    output_file_path = tmp_path / output_filename

    success = load_to_csv(sample_transformed_df, str(output_file_path))

    assert success is True, "load_to_csv seharusnya mengembalikan True untuk operasi yang sukses."
    assert output_file_path.exists(), "File CSV seharusnya sudah dibuat."

    # Verifikasi konten file CSV
    df_read = pd.read_csv(output_file_path)
    # Membandingkan DataFrame, check_dtype=False karena tipe bisa berubah saat baca CSV
    pd.testing.assert_frame_equal(df_read, sample_transformed_df, check_dtype=False)

    assert len(df_read) == len(sample_transformed_df), "Jumlah baris di CSV tidak sesuai."
    assert list(df_read.columns) == list(sample_transformed_df.columns), "Kolom di CSV tidak sesuai."

def test_load_to_csv_empty_dataframe(tmp_path):
    """Tes bagaimana load_to_csv menangani DataFrame kosong."""
    empty_df = pd.DataFrame()
    output_filename = "empty_test_products.csv"
    output_file_path = tmp_path / output_filename

    success = load_to_csv(empty_df, str(output_file_path))

    assert success is False, "load_to_csv seharusnya mengembalikan False untuk DataFrame kosong."
    assert not output_file_path.exists(), "File CSV seharusnya tidak dibuat untuk DataFrame kosong."

def test_load_to_csv_none_dataframe(tmp_path):
    """Tes bagaimana load_to_csv menangani DataFrame None."""
    none_df = None
    output_filename = "none_test_products.csv"
    output_file_path = tmp_path / output_filename

    success = load_to_csv(none_df, str(output_file_path))

    assert success is False, "load_to_csv seharusnya mengembalikan False untuk DataFrame None."
    assert not output_file_path.exists(), "File CSV seharusnya tidak dibuat untuk DataFrame None."

def test_load_to_csv_with_subdirectories(sample_transformed_df, tmp_path):
    output_dir = tmp_path / "data_output" / "files"
    output_filename = "products_in_subdir.csv"
    output_file_path = output_dir / output_filename

    success = load_to_csv(sample_transformed_df, str(output_file_path))

    assert success is True, "load_to_csv gagal dengan path subdirektori."
    assert output_file_path.exists(), "File CSV di subdirektori seharusnya sudah dibuat."
    
    df_read = pd.read_csv(output_file_path)
    pd.testing.assert_frame_equal(df_read, sample_transformed_df, check_dtype=False)