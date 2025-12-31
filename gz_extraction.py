import tarfile
import glob
import os

input_pattern = '/path/to/downloaded/images'  
destination_root = '/destination/path'

for tar_path in glob.glob(input_pattern):
    # e.g., tar_path = "images_01.tar.gz"
    base_name = os.path.splitext(os.path.splitext(os.path.basename(tar_path))[0])[0]
    # This splits "images_01.tar.gz" into ("images_01.tar", ".gz"), then again into ("images_01", ".tar")
    
    extract_dir = os.path.join(destination_root, base_name)
    os.makedirs(extract_dir, exist_ok=True)
    
    print(f"Extracting {tar_path} to {extract_dir}")
    with tarfile.open(tar_path, "r:gz") as tar_ref:
        tar_ref.extractall(path=extract_dir)

# The above code extracts all the tar.gz files in the current directory to the destination_root directory.
# The extracted files will be in the format "images_01",
# where 01 is the number of the tar.gz file that was extracted.