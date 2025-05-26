import pandas as pd
import os

def load_to_csv(df, output_path, **kwargs):
    if df is None or df.empty:
        print("DataFrame kosong, tidak ada data untuk disimpan ke CSV.")
        return False
    try:
        # Dapatkan path direktori dari output_path
        directory = os.path.dirname(output_path)

        if directory and not os.path.exists(directory): #
            os.makedirs(directory) #
            print(f"Direktori dibuat: {directory}")

        df.to_csv(output_path, index=False, **kwargs)
        print(f"Data berhasil disimpan ke CSV: {output_path}")
        return True
    except Exception as e:
        print(f"Gagal menyimpan data ke CSV di {output_path}: {e}")
        return False