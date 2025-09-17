import os
import requests
import hashlib
from urllib.parse import urlparse

SAVE_DIR = "downloads"
os.makedirs(SAVE_DIR, exist_ok=True)

downloaded_hashes = set()

SAFE_CONTENT_TYPES = {"image/jpeg", "image/png", "image/gif"}

def download_file(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10, stream=True)

        if response.status_code != 200:
            print(f" Skipped {url} (status {response.status_code})")
            return

        content_type = response.headers.get("Content-Type", "").split(";")[0].strip()
        if content_type not in SAFE_CONTENT_TYPES:
            print(f" Skipped {url} (unsupported content type: {content_type})")
            return

        filename = os.path.basename(urlparse(url).path)
        if not filename:
            filename = "downloaded_file"

        filepath = os.path.join(SAVE_DIR, filename)

        file_hash = hashlib.sha256()
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(8192):
                f.write(chunk)
                file_hash.update(chunk)

        hash_hex = file_hash.hexdigest()
        if hash_hex in downloaded_hashes:
            os.remove(filepath)  
            print(f" Duplicate skipped: {filename}")
        else:
            downloaded_hashes.add(hash_hex)
            print(f" Downloaded: {filename}")

    except requests.exceptions.RequestException as e:
        print(f" Error downloading {url}: {e}")


def download_multiple(urls):
    for url in urls:
        download_file(url)


urls = [
    "https://example.com/image1.jpg",
    "https://example.com/image2.png",
    "https://example.com/not_an_image.exe"
]

download_multiple(urls)
