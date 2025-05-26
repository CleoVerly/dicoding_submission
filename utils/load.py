import pandas as pd
import os

def load_to_csv(df, output_path, **kwargs):
    """
    Menyimpan DataFrame ke file CSV.
    output_path adalah path lengkap ke file CSV output.
    """
    if df is None or df.empty:
        print("DataFrame kosong, tidak ada data untuk disimpan ke CSV.")
        return False
    
    try:
        # Dapatkan path direktori dari output_path
        directory = os.path.dirname(output_path)
        
        # Hanya buat direktori jika 'directory' tidak kosong (artinya output_path mengandung path folder)
        if directory and not os.path.exists(directory): #
            os.makedirs(directory) #
            print(f"Direktori dibuat: {directory}")
        
        df.to_csv(output_path, index=False, **kwargs)
        print(f"Data berhasil disimpan ke CSV: {output_path}")
        return True
    except Exception as e:
        print(f"Gagal menyimpan data ke CSV di {output_path}: {e}")
        return False

# --- Kerangka untuk Fungsi Load Tambahan (Opsional untuk Kriteria 2) ---

# def load_to_google_sheets(df, sheet_name, credentials_path='google-sheets-api.json'):
#     """Menyimpan DataFrame ke Google Sheets."""
#     if df is None or df.empty:
#         print("DataFrame kosong, tidak ada data untuk disimpan ke Google Sheets.")
#         return False
#     try:
#         # import gspread
#         # from google.oauth2.service_account import Credentials
#         #
#         # scopes = ['https://www.googleapis.com/auth/spreadsheets']
#         # credentials = Credentials.from_service_account_file(credentials_path, scopes=scopes)
#         # gc = gspread.authorize(credentials)
#         #
#         # spreadsheet = gc.open(sheet_name) # Ganti dengan nama Spreadsheet Anda
#         # worksheet = spreadsheet.sheet1 
#         #
#         # worksheet.clear() # Hapus data lama (opsional)
#         # worksheet.update([df.columns.values.tolist()] + df.values.tolist()) # Tulis header dan data
#         print(f"Data berhasil disimpan ke Google Sheets: {sheet_name}")
#         return True
#     except Exception as e:
#         print(f"Gagal menyimpan data ke Google Sheets {sheet_name}: {e}")
#         return False

# def load_to_postgresql(df, table_name, db_params):
#     """Menyimpan DataFrame ke tabel PostgreSQL."""
#     if df is None or df.empty:
#         print("DataFrame kosong, tidak ada data untuk disimpan ke PostgreSQL.")
#         return False
#     try:
#         # from sqlalchemy import create_engine
#         #
#         # conn_string = f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}"
#         # engine = create_engine(conn_string)
#         #
#         # df.to_sql(table_name, engine, if_exists='replace', index=False) # 'replace', 'append', atau 'fail'
#         print(f"Data berhasil disimpan ke tabel PostgreSQL: {table_name}")
#         return True
#     except Exception as e:
#         print(f"Gagal menyimpan data ke PostgreSQL tabel {table_name}: {e}")
#         return False