import os
import requests
import secrets
import concurrent.futures

def generate_random_string(length):
    return secrets.token_hex(length)

def download_image(img_url, folder_name):
    img_ext = os.path.splitext(img_url)[1]
    img_name = generate_random_string(6) + img_ext
    img_path = os.path.join(folder_name, img_name)

    img_response = requests.get(img_url)
    with open(img_path, "wb") as img_file:
        img_file.write(img_response.content)
    print(f"Downloaded {img_url} to {img_path}")

# Create a folder to save the images
folder_name = "data/Spider"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# iNaturalist API request parameters
api_url = "https://api.inaturalist.org/v1/observations"
params = {
    "taxon_id": 47118,  # Correct Ant taxon ID
    "per_page": 200,  # Maximum allowed per request is 200
    "page": 1,  # Starting page
    "quality_grade": "research",  # Filter by research-grade images
}

while True:
    # Send API request
    response = requests.get(api_url, params=params)
    data = response.json()
    observations = data["results"]

    # If no observations found, break the loop
    if not observations:
        break

    # Download and save the images
    img_urls = []
    for observation in observations:
        if "taxon" in observation and "photos" in observation:
            for photo in observation["photos"]:
                img_urls.append(photo["url"])

    # Use 4 threads to download images
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(download_image, img_url, folder_name) for img_url in img_urls]
        concurrent.futures.wait(futures)

    # Go to the next page
    params["page"] += 1
