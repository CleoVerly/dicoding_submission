import pytest
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from unittest.mock import patch, MagicMock, ANY
import requests

from utils.extract import scrape_page, extract_product_details, extract_data_from_website

# --- Tes untuk scrape_page ---

@patch('utils.extract.requests.get')
def test_scrape_page_success(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = "<html><body><h1>Test Page</h1></body></html>"
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    soup = scrape_page("http://example.com/success")
    
    assert soup is not None
    assert soup.find('h1').text == "Test Page"
    mock_get.assert_called_once_with("http://example.com/success", headers=ANY, timeout=ANY)

@patch('utils.extract.requests.get')
def test_scrape_page_request_exception(mock_get): # Ganti nama agar lebih jelas dari test_scrape_page_failure
    mock_get.side_effect = requests.exceptions.RequestException("Simulated Network Error")
    soup = scrape_page("http://example.com/network_error")
    assert soup is None
    mock_get.assert_called_once_with("http://example.com/network_error", headers=ANY, timeout=ANY)

@patch('utils.extract.requests.get')
def test_scrape_page_timeout(mock_get):
    mock_get.side_effect = requests.exceptions.Timeout("Simulated Timeout")
    soup = scrape_page("http://example.com/timeout_error")
    assert soup is None
    mock_get.assert_called_once_with("http://example.com/timeout_error", headers=ANY, timeout=ANY)

@patch('utils.extract.requests.get')
def test_scrape_page_http_error(mock_get):
    mock_response = MagicMock()
    # Simulasikan raise_for_status() yang memunculkan HTTPError
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Simulated HTTP Error 404")
    mock_get.return_value = mock_response
    
    soup = scrape_page("http://example.com/http_404_error")
    assert soup is None
    mock_get.assert_called_once_with("http://example.com/http_404_error", headers=ANY, timeout=ANY)

# --- Tes untuk extract_product_details ---

def test_extract_product_details_valid():
    html_card_content = """
    <div class="collection-card">
        <div class="product-details">
            <h3 class="product-title">Awesome T-Shirt</h3>
            <div class="price-container"><span class="price">$29.99</span></div>
            <p>Rating: ⭐ 4.5 / 5</p>
            <p>3 Colors</p>
            <p>Size: L</p>
            <p>Gender: Unisex</p>
        </div>
    </div>
    """
    card_soup = BeautifulSoup(html_card_content, 'html.parser').find('div', class_='collection-card')
    details = extract_product_details(card_soup)

    assert details is not None
    assert details['title'] == "Awesome T-Shirt"
    assert details['price'] == "$29.99"
    assert "Rating: ⭐ 4.5 / 5" in details['rating'] 
    assert "3 Colors" in details['colors']
    assert "Size: L" in details['size']
    assert "Gender: Unisex" in details['gender']

def test_extract_product_details_missing_all_optional_info():
    html_card_content_missing = """
    <div class="collection-card">
        <div class="product-details">
            <h3 class="product-title">Basic Shirt</h3>
            </div>
    </div>
    """
    card_soup = BeautifulSoup(html_card_content_missing, 'html.parser').find('div', class_='collection-card')
    details = extract_product_details(card_soup)

    assert details is not None
    assert details['title'] == "Basic Shirt"
    assert details['price'] is None 
    assert details['rating'] is None
    assert details['colors'] is None
    assert details['size'] is None
    assert details['gender'] is None

def test_extract_product_details_no_product_details_div():
    # Skenario jika div 'product-details' tidak ada
    html_card_no_details = """
    <div class="collection-card">
        <h3 class="product-title-outside">Shirt No Details Div</h3>
    </div>
    """
    card_soup = BeautifulSoup(html_card_no_details, 'html.parser').find('div', class_='collection-card')
    details = extract_product_details(card_soup)
    assert details is None # Sesuai implementasi jika product_details_div tidak ditemukan

def test_extract_product_details_no_title_tag():
    # Skenario jika h3 'product-title' tidak ada
    html_card_no_title = """
    <div class="collection-card">
        <div class="product-details">
            <div class="price-container"><span class="price">$29.99</span></div>
        </div>
    </div>
    """
    card_soup = BeautifulSoup(html_card_no_title, 'html.parser').find('div', class_='collection-card')
    details = extract_product_details(card_soup)
    # Asumsi jika tidak ada judul, item tidak valid
    assert details is None 

def test_extract_product_details_price_unavailable_text():
    # Skenario jika harga adalah teks "Price Unavailable"
    html_card_price_unavailable = """
    <div class="collection-card">
        <div class="product-details">
            <h3 class="product-title">Unavailable Price Shirt</h3>
            <p class="price">Price Unavailable</p> 
        </div>
    </div>
    """
    card_soup = BeautifulSoup(html_card_price_unavailable, 'html.parser').find('div', class_='collection-card')
    details = extract_product_details(card_soup)
    assert details is not None
    assert details['price'] == "Price Unavailable"


# --- Tes untuk extract_data_from_website ---

@patch('utils.extract.scrape_page')
@patch('utils.extract.extract_product_details')
def test_extract_data_from_website_successful_run(mock_extract_details, mock_scrape_page):
    mock_soup_page1 = MagicMock(spec=BeautifulSoup)
    mock_card1 = BeautifulSoup('<div class="collection-card">Card1</div>', 'html.parser').div
    mock_card2 = BeautifulSoup('<div class="collection-card">Card2</div>', 'html.parser').div
    mock_soup_page1.find_all.return_value = [mock_card1, mock_card2]

    mock_soup_page2 = MagicMock(spec=BeautifulSoup) # Halaman kedua kosong
    mock_soup_page2.find_all.return_value = []

    mock_scrape_page.side_effect = [mock_soup_page1, mock_soup_page2] # Dua halaman disimulasikan

    mock_extract_details.side_effect = [
        {"title": "Product 1", "price": "$10", "rating": "5/5", "colors": "1 Color", "size": "S", "gender": "Men"},
        {"title": "Product 2", "price": "$20", "rating": "4/5", "colors": "2 Colors", "size": "M", "gender": "Women"}
    ]

    with patch('utils.extract.MAX_PAGES', 2):
        df = extract_data_from_website()

    assert not df.empty
    assert len(df) == 2
    assert 'timestamp' in df.columns
    assert df['title'].tolist() == ["Product 1", "Product 2"]
    assert mock_scrape_page.call_count == 2
    assert mock_extract_details.call_count == 2

@patch('utils.extract.scrape_page')
def test_extract_data_from_website_all_pages_fail_to_scrape(mock_scrape_page):
    mock_scrape_page.return_value = None

    with patch('utils.extract.MAX_PAGES', 3):
        df = extract_data_from_website()

    assert df.empty
    assert mock_scrape_page.call_count == 3

@patch('utils.extract.scrape_page')
def test_extract_data_from_website_no_product_cards_on_any_page(mock_scrape_page):
    mock_soup = MagicMock(spec=BeautifulSoup)
    mock_soup.find_all.return_value = []
    mock_scrape_page.return_value = mock_soup

    with patch('utils.extract.MAX_PAGES', 2):
        df = extract_data_from_website()

    assert df.empty
    assert mock_scrape_page.call_count == 2

@patch('utils.extract.scrape_page')
@patch('utils.extract.extract_product_details')
def test_extract_data_from_website_all_details_extraction_fail(mock_extract_details, mock_scrape_page):
    mock_soup = MagicMock(spec=BeautifulSoup)
    mock_card = BeautifulSoup('<div class="collection-card">Card</div>', 'html.parser').div
    mock_soup.find_all.return_value = [mock_card, mock_card] # Ada kartu produk
    mock_scrape_page.return_value = mock_soup

    mock_extract_details.return_value = None # Ekstraksi detail selalu gagal

    with patch('utils.extract.MAX_PAGES', 1):
        df = extract_data_from_website()

    assert df.empty
    assert mock_extract_details.call_count == 2

@patch('utils.extract.scrape_page')
@patch('utils.extract.extract_product_details')
def test_extract_data_from_website_partial_scrape_and_extract_success(mock_extract_details, mock_scrape_page):

    mock_soup_page1 = MagicMock(spec=BeautifulSoup)
    mock_card1_p1 = BeautifulSoup('<div class="collection-card">P1C1</div>', 'html.parser').div
    mock_card2_p1 = BeautifulSoup('<div class="collection-card">P1C2</div>', 'html.parser').div
    mock_soup_page1.find_all.return_value = [mock_card1_p1, mock_card2_p1]
    
    mock_soup_page3 = MagicMock(spec=BeautifulSoup)
    mock_card1_p3 = BeautifulSoup('<div class="collection-card">P3C1</div>', 'html.parser').div
    mock_soup_page3.find_all.return_value = [mock_card1_p3]

    mock_scrape_page.side_effect = [mock_soup_page1, None, mock_soup_page3]

    product1_details = {"title": "Prod1", "price": "$1", "rating": "1/5", "colors": "1C", "size": "S", "gender": "M"}
    product3_details = {"title": "Prod3", "price": "$3", "rating": "3/5", "colors": "3C", "size": "L", "gender": "F"}
    mock_extract_details.side_effect = [product1_details, None, product3_details] # Detail kedua gagal

    with patch('utils.extract.MAX_PAGES', 3):
        df = extract_data_from_website()
        
    assert len(df) == 2
    assert df['title'].tolist() == ["Prod1", "Prod3"]
    assert mock_scrape_page.call_count == 3
    assert mock_extract_details.call_count == 3
    
    # Cek timestamp ada di semua baris yang berhasil
    assert 'timestamp' in df.columns
    assert df['timestamp'].notna().all()