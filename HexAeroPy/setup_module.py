import os
import zipfile
import requests
from tqdm import tqdm

def download_and_extract_zip(url, extract_to):
    """
    Downloads a ZIP file from a given URL and extracts its contents to a specified directory.
    Adds a loading bar to show download progress and prints additional information about the download.

    Parameters:
    - url (str): The URL of the ZIP file to download.
    - extract_to (str): The directory path where the contents of the ZIP file should be extracted.
    """
    
    # Ensure the extract_to directory exists
    if not os.path.exists(extract_to):
        print(f"\nCreating directory {extract_to}...")
        os.makedirs(extract_to, exist_ok=True)
    
    print(f"\nDownloading data from {url}...")
    local_zip_path = os.path.join(extract_to, os.path.basename(url))
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

    with open(local_zip_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")

    print(f"Extracting data to {extract_to}...")
    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(local_zip_path)
    print("Download and extraction complete.")


def setup_data_directory(base_dir):
    """
    Sets up the data directory by downloading and extracting necessary datasets.
    """
    urls = [
        "https://zenodo.org/records/10651018/files/airport_hex.zip?download=1",
        "https://zenodo.org/records/10729365/files/runway_hex.zip?download=1", 
        "https://zenodo.org/records/10651018/files/test_data.zip?download=1"
        #"https://www.eurocontrol.int/performance/data/download/other/hexaero/airport_hex.zip",
        #"https://www.eurocontrol.int/performance/data/download/other/hexaero/runway_hex.zip",
        #"https://www.eurocontrol.int/performance/data/download/other/hexaero/test_data.zip"
    ]
    for url in urls:
        download_and_extract_zip(url, base_dir)

def ensure_data_available(auto_install = False, local_install = False):
    """
    Ensure that the necessary data files are available, and download them if not.
    This function asks for user permission before downloading the data.
    """
    
    if local_install:
        data_dir = os.path.join('..', 'data')
    else: 
        data_dir = os.path.join(__file__.replace('/setup_module.py', ''), 'data')
    required_folders = ['airport_hex', 'runway_hex', 'test_data']  # Adjusted for parquet files

    if not all(os.path.exists(os.path.join(data_dir, f)) for f in required_folders):
        if auto_install:
            print("[Downloading required data files...]")
            setup_data_directory(data_dir)
        else:
            user_response = input("Required metadata parquet files not found. Download (~600MB) and setup now? [y/n]: ")
            if user_response.lower() == 'y':
                print("\n[Downloading required data files...]")
                setup_data_directory(data_dir)
            else:
                print("Data download skipped. The package requires data files to function properly.")