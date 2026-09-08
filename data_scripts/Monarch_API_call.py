
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
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" # using path like this resolves it to <repo root>/data/Monarch_images
# and helps with reproducbility - now this can run on any machine
MONARCH_DIR = DATA_DIR / "Monarch_images"

API_URL = "https://api.inaturalist.org/v1/observations"
TAXON_ID = 48662
NICKNAME = "Monarch"
START_PAGE = 51
NUM_PAGES = 2
PER_PAGE = 200
PLACE_ID = 97394
MONTHS = "6,7"
QUALITY_GRADE = "research"
REQUEST_DELAY_SECONDS = 2
METADATA_OUTPUT_FILENAME = f"{NICKNAME}_metadata_testset.json"

def query_taxa(
    taxon_id=TAXON_ID,
    start_page=START_PAGE,
    num_pages=NUM_PAGES,
    per_page=PER_PAGE,
    place_id=PLACE_ID,
    months=MONTHS,
    quality_grade=QUALITY_GRADE,
    request_delay_seconds=REQUEST_DELAY_SECONDS,
):

    params = {
        "taxon_id": taxon_id,
        "per_page": per_page,
        "page": 1,
        "quality_grade": quality_grade,
        "place_id": place_id,
        "month": months,
    }

    observations = []  # Accumulate all results here

    #while True:
    for page in range(start_page, start_page + num_pages):
        print(f"Fetching page {page}...")
        params["page"] = page

        try:
            response = requests.get(API_URL, params=params)
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
            time.sleep(request_delay_seconds)  # Increase delay to avoid getting blocked

        except requests.exceptions.HTTPError as err:
            print(f"HTTP Error: {err}")
            if response.status_code == 403:
                print("403 Forbidden: Possible rate limit reached. Try waiting longer between requests or using authentication.")
            break  # Stop fetching if API blocks us

    return observations  # Return all accumulated research-grade observations


if __name__ == "__main__":
    obs_data = query_taxa()

    os.makedirs(MONARCH_DIR, exist_ok=True)
    filename = MONARCH_DIR / METADATA_OUTPUT_FILENAME

    with open(filename, "w") as f:
        json.dump({"total_results": len(obs_data), "results": obs_data}, f, indent=2)

    print(f"Saved {len(obs_data)} research-grade observations to {filename}")
