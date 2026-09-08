

"""
Maximum download of 5 GB of media per hour or 
24 GB of media per day may result in a permanent block

will download all the first observation_photos in each observation in the json stored in API_jsons folder

get the jsons by running Monarch_API_call.py

run using Monarch_API_download.sh

image ids and class is saved to Monarch_images.csv in the butterfly folder - append each time


"""

import json
import requests
import os
from csv import writer

def download_images_from_json(metadata, download_dir):

	# Create download directory if it doesn't exist.
    os.makedirs(download_dir, exist_ok=True)

    with open(metadata, "r") as f:
        data = json.load(f)
    
    # Loop through the taxa (or observations) results.
    for observation in data.get("results", []):
        #if taxon.get("quality_grade") != "research":
        #    continue
 
        #default_photo = taxon.get("default_photo")
        #if not default_photo:
        #    continue  # Skip if there's no default photo.
        
        # Try to get at least a medium sized image.
        #image_url = (default_photo.get("medium_url") or 
        #             default_photo.get("large_url") or 
        #             default_photo.get("original_url"))

        #for photo in observation.get("observation_photos", []):
            #image_url = photo.get("photo", {}).get("url")
        observation_photos = observation.get("observation_photos", [])
        if observation_photos:  # Ensure there is at least one photo
            photo = observation_photos[0]  # Select only the first photo
            image_url = photo.get("photo", {}).get("url", "").replace("square", "medium")
            photo_id = photo.get("photo", {}).get("id")
            

            # Ensure the URL is from the AWS open data domain.
            if image_url and "inaturalist-open-data.s3.amazonaws.com" in image_url:
                #photo_id = default_photo.get("id", "unknown")
                
                print(f"Downloading photo ID {photo_id} from {image_url} of {nickname}")
                try:
                    img_response = requests.get(image_url, stream=True)
                    img_response.raise_for_status()
        
                    file_path = os.path.join(download_dir, f"{photo_id}.jpg")
                    file_name = os.path.join(f"{photo_id}.jpg")
                    with open(file_path, "wb") as img_file:
                        for chunk in img_response.iter_content(chunk_size=8192):
                            if chunk:
                                img_file.write(chunk)
                    print(f"Saved image as {file_path}")
                    with open('Monarch_images.csv', 'a') as f_object:
                        writer_object = writer(f_object)
                        Img_ID = [file_name, nickname]
                        writer_object.writerow(Img_ID)
                        f_object.close()

                except Exception as e:
                    print(f"Failed to download photo ID {photo_id}: {e}")

#edit this line every time
nickname = "Monarch"

if __name__ == "__main__":
	download_dir = 'Monarch_images/'
	metadata = 'Monarch_images/Monarch_metadata.json' #EDIT
	download_images_from_json(metadata, download_dir)
