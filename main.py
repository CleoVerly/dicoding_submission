from utils.extract import extract_data_from_website
from utils.transform import transform_data
from utils.load import load_to_csv

# output CSV
CSV_OUTPUT_FILENAME = "products.csv" # [cite: 16]

# (Opsional) Konfigurasi untuk Google Sheets atau PostgreSQL jika digunakan
# GOOGLE_SHEET_NAME = "Nama Spreadsheet Anda"
# GOOGLE_SHEETS_CREDENTIALS = "google-sheets-api.json" # [cite: 16]

# DB_PARAMS = {
# "user": "your_user",
# "password": "your_password",
# "host": "localhost",
# "port": "5432",
# "dbname": "your_database"
# }
# POSTGRES_TABLE_NAME = "products_fashion_studio"

def main_pipeline():
    print("Memulai ETL Pipeline...")

    # 1. Tahap Ekstraksi
    print("\n--- Tahap Ekstraksi Dimulai ---")
    raw_product_data = extract_data_from_website()
    if raw_product_data is None or raw_product_data.empty:
        print("Ekstraksi gagal atau tidak menghasilkan data. Pipeline dihentikan.")
        return
    print(f"Ekstraksi selesai. Jumlah data mentah: {len(raw_product_data)}")

    # 2. Tahap Transformasi
    print("\n--- Tahap Transformasi Dimulai ---")
    transformed_product_data = transform_data(raw_product_data)
    if transformed_product_data is None or transformed_product_data.empty:
        print("Transformasi gagal atau tidak menghasilkan data bersih. Pipeline dihentikan.")
        return
    print(f"Transformasi selesai. Jumlah data bersih: {len(transformed_product_data)}")
    print("Sample data setelah transformasi:")
    print(transformed_product_data.head())

    # 3. Tahap Load
    print("\n--- Tahap Load Dimulai ---")
    # Menyimpan ke CSV (Basic Requirement)
    csv_success = load_to_csv(transformed_product_data, CSV_OUTPUT_FILENAME)
    if csv_success:
        print(f"Data berhasil dimuat ke {CSV_OUTPUT_FILENAME}")
    else:
        print(f"Gagal memuat data ke {CSV_OUTPUT_FILENAME}")

    # (Opsional) Menyimpan ke Google Sheets (Skilled/Advanced Requirement)
    # if csv_success: # Mungkin Anda hanya ingin load ke GSheets jika CSV berhasil
    # print("\n--- Memuat ke Google Sheets ---")
    # gsheets_success = load_to_google_sheets(transformed_product_data, GOOGLE_SHEET_NAME, GOOGLE_SHEETS_CREDENTIALS)
    # if gsheets_success:
    # print(f"Data berhasil dimuat ke Google Sheets: {GOOGLE_SHEET_NAME}")
    # else:
    # print(f"Gagal memuat data ke Google Sheets: {GOOGLE_SHEET_NAME}")

    # (Opsional) Menyimpan ke PostgreSQL (Skilled/Advanced Requirement)
    # if csv_success: # Mungkin Anda hanya ingin load ke DB jika CSV berhasil
    # print("\n--- Memuat ke PostgreSQL ---")
    # postgres_success = load_to_postgresql(transformed_product_data, POSTGRES_TABLE_NAME, DB_PARAMS)
    # if postgres_success:
    # print(f"Data berhasil dimuat ke PostgreSQL tabel: {POSTGRES_TABLE_NAME}")
    # else:
    # print(f"Gagal memuat data ke PostgreSQL tabel: {POSTGRES_TABLE_NAME}")


    print("\nETL Pipeline Selesai.")

if __name__ == '__main__':
    main_pipeline()