
"""
For monarch images - loop version gets a lot more photos
"""

"""

This code will download 50 pages (each with 200 results for a total of 10000 observations) from the iNaturalist API into a json
Jsons will be stored in the folder API_jsons in butterflyAI

Observations are filtered for research grade before being downloaded

Change the taxon_id and nickname each time you run the code for a new species

run butterflyAI_image_download.py to download images stored in the json downloaded here

"""


import requests
import json
import os
import time

def query_taxa(TAXON_ID, nickname):
    api_url = "https://api.inaturalist.org/v1/observations"
    params = {
        "taxon_id": TAXON_ID,
        "per_page": 200,
        "page": 1,
        "quality_grade": "research", 
        "place_id": 97394,   
        "month": "6,7" 
    }
    
    page = 51
    observations = []  # Accumulate all results here

    start_page = 51
    num_pages = 2

    #while True:
    for page in range(start_page, start_page + num_pages):
        print(f"Fetching page {page}...")
        params["page"] = page

        try:
            response = requests.get(api_url, params=params)
            response.raise_for_status()  # Raise error for bad response

            data = response.json()
            results = data.get("results", [])
            total_results = data.get("total_results", 0)
            
            print(f"Total observations available: {total_results}")
            print(f"Observations found on this page: {len(results)}")

            # Stop if no more results
            if not results:
                print("No more observations found.")
                break

            # Append research-grade results to observations
            observations.extend([obs for obs in results if obs.get("quality_grade") == "research"])

            print("One loop done")

            # Move to the next page
            page += 1
            time.sleep(2)  # Increase delay to avoid getting blocked

        except requests.exceptions.HTTPError as err:
            print(f"HTTP Error: {err}")
            if response.status_code == 403:
                print("403 Forbidden: Possible rate limit reached. Try waiting longer between requests or using authentication.")
            break  # Stop fetching if API blocks us

    return observations  # Return all accumulated research-grade observations


if __name__ == "__main__":
    TAXON_ID = 48662
    nickname = 'Monarch'
    obs_data = query_taxa(TAXON_ID, nickname)  # Now accumulates all pages

    folder = "Monarch_images_test_set"
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, nickname + '_metadata_testset.json')

    # Save all collected observations to the JSON file
    with open(filename, "w") as f:
        json.dump({"total_results": len(obs_data), "results": obs_data}, f, indent=2)

    print(f"Saved {len(obs_data)} research-grade observations to {filename}")
