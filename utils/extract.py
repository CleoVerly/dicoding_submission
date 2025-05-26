import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time

# BASE_URL
BASE_URL = "https://fashion-studio.dicoding.dev/"
MAX_PAGES = 50

def scrape_page(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup
    except requests.exceptions.Timeout:
        print(f"Timeout saat mengakses URL {url}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return None

def extract_product_details(product_card_soup):
    """
    Mengekstrak detail dari satu kartu produk (soup dari elemen <div class="collection-card">).
    """
    details = {
        "title": None,
        "price": None,
        "rating": None,
        "colors": None,
        "size": None,
        "gender": None
    }
    
    product_details_div = product_card_soup.find('div', class_='product-details')
    if not product_details_div:
        return None

    # Extract Title
    title_tag = product_details_div.find('h3', class_='product-title')
    if title_tag:
        details['title'] = title_tag.text.strip()
    else:
        return None

    # Extract Price
    price_container_tag = product_details_div.find('div', class_='price-container')
    if price_container_tag:
        price_tag = price_container_tag.find('span', class_='price')
        if price_tag:
            details['price'] = price_tag.text.strip()
    else:
        price_unavailable_tag = product_details_div.find('p', class_='price')
        if price_unavailable_tag and "Price Unavailable" in price_unavailable_tag.text:
            details['price'] = "Price Unavailable"
        elif product_details_div.find(string="Price Unavailable"): # Fallback jika tidak ada class 'price'
             details['price'] = "Price Unavailable"


    # Extract Rating, Colors, Size, Gender from <p> tags
    p_tags = product_details_div.find_all('p')
    for p_tag in p_tags:
        text = p_tag.text.strip()
        if "Rating:" in text:
            details['rating'] = text
        elif "Colors" in text and not "Rating:" in text and not "Size:" in text and not "Gender:" in text:
            details['colors'] = text
        elif "Size:" in text:
            details['size'] = text
        elif "Gender:" in text:
            details['gender'] = text

    return details

def extract_data_from_website():
    all_products_data = []
    # Tambah kolom timestamp untuk skor "Skilled"
    current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 

    print(f"Memulai ekstraksi data pada: {current_timestamp}")

    for page_num in range(1, MAX_PAGES + 1):
        if page_num == 1:
            page_url = BASE_URL
        else:
            page_url = f"{BASE_URL}page{page_num}"
        
        print(f"Scraping halaman: {page_url}")
        soup = scrape_page(page_url)

        if soup:
            # Container utama produk adalah <div class="collection-card">
            product_cards = soup.find_all('div', class_='collection-card') 

            if not product_cards and page_num > 1 :
                print(f"Tidak ada produk ditemukan di halaman {page_num}. Mungkin sudah mencapai halaman terakhir atau ada perubahan struktur.")

            for card_soup in product_cards:
                product_info = extract_product_details(card_soup)
                if product_info and product_info.get('title'):
                    product_info['timestamp'] = current_timestamp
                    all_products_data.append(product_info)
            
            time.sleep(0.5)

        else:
            print(f"Gagal mengambil data dari halaman {page_num}. Melanjutkan ke halaman berikutnya jika ada.")

    if not all_products_data:
        print("Tidak ada data produk yang berhasil diekstrak.")
        return pd.DataFrame()

    df_products = pd.DataFrame(all_products_data)
    print(f"Ekstraksi selesai. Total {len(df_products)} produk berhasil diambil.")
    
    expected_columns = ['title', 'price', 'rating', 'colors', 'size', 'gender', 'timestamp']
    
    final_df_columns = []
    for col in expected_columns:
        if col in df_products.columns:
            final_df_columns.append(col)
    
    df_products = df_products[final_df_columns]
    
    for col in expected_columns:
        if col not in df_products.columns:
            df_products[col] = None

    return df_products