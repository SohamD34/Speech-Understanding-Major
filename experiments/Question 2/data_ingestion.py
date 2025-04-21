import zipfile
import os

# Download the dataset ZIP files at data/denoising/
# - set1
# - set2

def unzip_dataset(zip_file_path, extract_to):
    """
    Unzips the dataset ZIP file to the specified directory.
    
    Args:
        zip_file_path (str): Path to the ZIP file.
        extract_to (str): Directory to extract the contents to.
    """
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"Extracted {zip_file_path} to {extract_to}")



if __name__ == "__main__":

    zip_file_paths = [
        "set 1 - Clean and noisy-20250418T102555Z-001.zip",
        "set 2 - only noisy-20250418T102559Z-001.zip"
    ]
    extract_path = 'data/denoising/'
    os.makedirs(extract_path, exist_ok=True)

    for i in range(2):

        zip_file_path =  extract_path + zip_file_paths[i]
        unzip_dataset(zip_file_path, extract_path)

        os.rename(extract_path + zip_file_paths[i], f'data/denoising/set {i+1}')

    print("All datasets unzipped successfully.")