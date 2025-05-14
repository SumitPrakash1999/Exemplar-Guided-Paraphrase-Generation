import requests
import zipfile
import os
import shutil

# File paths
glove_file_path = "glove/glove.6B.300d.txt"
zip_file_path = "glove.6B.zip"
temp_dir = "glove_temp"

# Step 1: Check if the GloVe file already exists
if not os.path.exists(glove_file_path):
    print("GloVe file not found. Downloading...")
    
    # Step 2: Download the GloVe zip file
    url = "http://nlp.stanford.edu/data/glove.6B.zip"
    response = requests.get(url, stream=True)
    
    # Step 3: Save the downloaded zip file
    with open(zip_file_path, "wb") as f:
        f.write(response.content)
    
    # Step 4: Unzip the downloaded file and extract only the 300d file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        for file in zip_ref.namelist():
            if file == "glove.6B.300d.txt":
                zip_ref.extract(file, temp_dir)
    
    # Step 5: Move the specific GloVe file to the desired location
    extracted_glove_file_path = os.path.join(temp_dir, "glove.6B.300d.txt")
    
    # Create the target directory if it doesn't exist
    os.makedirs(os.path.dirname(glove_file_path), exist_ok=True)
    
    # Move the GloVe file to the specified directory
    shutil.move(extracted_glove_file_path, glove_file_path)
    
    # Step 6: Clean up by removing the zip file and the temporary directory
    os.remove(zip_file_path)
    shutil.rmtree(temp_dir)

    print("GloVe embeddings downloaded and saved to:", glove_file_path)
else:
    print("GloVe embeddings already exist at:", glove_file_path)
