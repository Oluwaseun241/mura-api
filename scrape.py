import os
import time
import logging
import requests
from PIL import Image
from icrawler.builtin import GoogleImageCrawler

# Settings
MAX_RETRIES = 3
MIN_WIDTH = 500
MIN_HEIGHT = 400
MAX_IMAGES = 100
USE_PROXY = False  # Set to True if you want to use free proxy rotation

os.makedirs("images", exist_ok=True)

# Function to get a free proxy (basic)
def get_free_proxy():
    try:
        res = requests.get("https://www.proxy-list.download/api/v1/get?type=http")
        proxies = res.text.strip().split("\r\n")
        return proxies[0] if proxies else None
    except Exception as e:
        print("Proxy fetch failed:", e)
        return None

def delete_low_res_images(folder, min_width=MIN_WIDTH, min_height=MIN_HEIGHT):
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        try:
            with Image.open(path) as img:
                if img.width < min_width or img.height < min_height:
                    os.remove(path)
        except:
            os.remove(path)  # delete corrupt or unreadable images

# Function to scrape images for a single dish
def scrape_dish(dish, max_images=MAX_IMAGES):
    folder = f'images/{dish.replace(" ", "_")}'
    os.makedirs(folder, exist_ok=True)

    for attempt in range(MAX_RETRIES):
        try:
            print(f"\nScraping: {dish} (Attempt {attempt + 1})")

            crawler = GoogleImageCrawler(
                downloader_threads=4,
                storage={'root_dir': folder}
            )

            # ✅ Set proxy after creating crawler
            if USE_PROXY:
                proxy = get_free_proxy()
                if proxy:
                    print(f"Using proxy: {proxy}")
                    crawler.downloader.proxy = f"http://{proxy}"

            # ✅ Filters must be passed in crawl()
            crawler.crawl(
                keyword=dish + " West African food",
                max_num=max_images,
                filters={'size': 'large'}
            )

            delete_low_res_images(folder)
            return
        except Exception as e:
            logging.error(f"Error scraping {dish}: {e}")
            time.sleep(3)
    else:
        print(f"❌ Failed to scrape: {dish} after {MAX_RETRIES} retries")

with open('west_african_dishes.txt', 'r', encoding='utf-8') as f:
    dishes = [line.strip() for line in f if line.strip()]

# Scrape all dishes
for dish in dishes:
    scrape_dish(dish)
    time.sleep(1)  # delay

