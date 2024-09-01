import os
import urllib.request
import zipfile
import tarfile

DATASET_FOLDER = "datasets"

DATASETS = {
    "skodar": "http://har-dataset.org/lib/exe/fetch.php?media=wiki:dataset:skodaminicp:skodaminicp_2015_08.zip",
    "wisdm": "https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz"
}

def download_and_extract(name, url, folder):
    # Create the main dataset folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    
    # Define the path for the downloaded file
    file_path = os.path.join(folder, f"{name}.zip")
    if url.endswith('.tar.gz'):
        file_path = os.path.join(folder, f"{name}.tar.gz")
    
    # Define the final extraction folder named as per the key in the DATASETS dictionary
    extract_folder = os.path.join(folder, name)
    
    # Create the extraction folder if it doesn't exist
    os.makedirs(extract_folder, exist_ok=True)
    
    # Download the file
    print(f"Downloading {name} from {url}...")
    urllib.request.urlretrieve(url, file_path)
    print(f"Downloaded {name} to {file_path}")
    
    # Extract the file based on its extension
    if file_path.endswith('.zip'):
        print(f"Extracting {name} from ZIP file to {extract_folder}...")
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
        print(f"Extracted {name} from ZIP file to {extract_folder}")
    elif file_path.endswith('.tar.gz'):
        print(f"Extracting {name} from TAR.GZ file to {extract_folder}...")
        with tarfile.open(file_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_folder)
        print(f"Extracted {name} from TAR.GZ file to {extract_folder}")
    else:
        print(f"Unsupported file type for {file_path}")
        return
    
    # Optionally, delete the downloaded file after extraction
    os.remove(file_path)
    print(f"Removed downloaded file {file_path}")

if __name__ == "__main__":

    for name, url in DATASETS.items():
        download_and_extract(name, url, DATASET_FOLDER)
