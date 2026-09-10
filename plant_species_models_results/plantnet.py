"""
Identify local flower photos with the Pl@ntNet API.

Outputs are written into plantnet_api_data_tests/plantnet_results by default.
"""

import argparse
import csv
import json
import mimetypes
import os
import re
import time
from pathlib import Path
from dotenv import load_dotenv

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "plantnet_api_data_tests"
DEFAULT_RESULTS_DIR = DEFAULT_IMAGE_DIR / "plantnet_results"
DEFAULT_OUTPUT_CSV = DEFAULT_RESULTS_DIR / "plantnet_identifications.csv"
DEFAULT_REPORT_PATH = DEFAULT_RESULTS_DIR / "plantnet_accuracy_report.txt"
DEFAULT_RAW_DIR = DEFAULT_RESULTS_DIR / "raw_json"
DOTENV_PATH = PROJECT_ROOT / ".env"

API_BASE_URL = "https://my-api.plantnet.org/v2/identify"
DEFAULT_PROJECT = "all"
DEFAULT_ORGAN = "flower"
DEFAULT_LANGUAGE = "en"
DEFAULT_NB_RESULTS = 5
DEFAULT_REQUEST_DELAY_SECONDS = 1.0
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png"}
OUTPUT_FIELDNAMES = [
    "FileName",
    "expected_label",
    "status",
    "error",
    "bestMatch",
    "top_score",
    "top_scientific_name",
    "top_scientific_name_authorship",
    "top_full_scientific_name",
    "top_genus",
    "top_family",
    "top_common_names",
    "predicted_organ",
    "predicted_organ_score",
    "plantnet_version",
    "remainingIdentificationRequests",
    "raw_json",
    "is_correct",
    "match_basis",
]

# create a new kind of exception called PlantNetRequestError
# allows us to clearly see api issues compared to other issues
class PlantNetRequestError(Exception):
    pass

def find_images(image_dir): # return a sorted list of JPG/JPEG/PNG files in this folder
    image_paths = []

    for path in image_dir.iterdir():
        is_supported_image = path.suffix.lower() in SUPPORTED_SUFFIXES #is it one of {".jpg", ".jpeg", ".png"}

        if path.is_file() and is_supported_image: # no subfolders sneaking their way in
            image_paths.append(path)

    return sorted(image_paths)


def clean_text(text):
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def expected_label_from_filename(image_path):
    return image_path.stem.replace("_", " ")


def score_prediction(expected_label, row):
    expected_words = clean_text(expected_label).split()
    predicted_text = clean_text(
        " ".join(
            [
                row["bestMatch"],
                row["top_scientific_name"],
                row["top_full_scientific_name"],
                row["top_common_names"],
            ]
        )
    )

    if expected_words and all(word in predicted_text for word in expected_words):
        return True, "all expected filename words found in PlantNet top result"

    if expected_words and any(word in predicted_text for word in expected_words):
        return False, "partial expected filename words found in PlantNet top result"

    return False, "no expected filename words found in PlantNet top result"


def identify_image(image_path, api_key, project, organ, language, nb_results):
    endpoint = f"{API_BASE_URL}/{project}"
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"

    with open(image_path, "rb") as image_file:
        files = [("images", (image_path.name, image_file, mime_type))]
        data = {"organs": [organ]}
        params = {
            "api-key": api_key,
            "lang": language,
            "nb-results": nb_results,
        }

        try:
            response = requests.post(endpoint, params=params, files=files, data=data, timeout=60)
        except requests.RequestException as error:
            raise PlantNetRequestError(type(error).__name__) from error

    if not response.ok:
        response_excerpt = response.text[:300].replace("\n", " ")
        message = f"HTTP {response.status_code}"
        if response_excerpt:
            message = f"{message}: {response_excerpt}"
        raise PlantNetRequestError(message)

    return response.json()


def get_nested(data, *keys, default=""):
    value = data
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return default if value is None else value


def summarize_result(image_path, result):
    top_result = result.get("results", [{}])[0] if result.get("results") else {}
    species = top_result.get("species", {})
    predicted_organ = result.get("predictedOrgans", [{}])[0] if result.get("predictedOrgans") else {}

    return {
        "FileName": image_path.name,
        "expected_label": expected_label_from_filename(image_path),
        "status": "ok",
        "error": "",
        "bestMatch": result.get("bestMatch", ""),
        "top_score": top_result.get("score", ""),
        "top_scientific_name": species.get("scientificNameWithoutAuthor", ""),
        "top_scientific_name_authorship": species.get("scientificNameAuthorship", ""),
        "top_full_scientific_name": species.get("scientificName", ""),
        "top_genus": get_nested(species, "genus", "scientificNameWithoutAuthor"),
        "top_family": get_nested(species, "family", "scientificNameWithoutAuthor"),
        "top_common_names": "; ".join(species.get("commonNames", [])),
        "predicted_organ": predicted_organ.get("organ", ""),
        "predicted_organ_score": predicted_organ.get("score", ""),
        "plantnet_version": result.get("version", ""),
        "remainingIdentificationRequests": result.get("remainingIdentificationRequests", ""),
        "raw_json": "",
        "is_correct": "",
        "match_basis": "",
    }


def summarize_error(image_path, error):
    row = dict.fromkeys(OUTPUT_FIELDNAMES, "")
    row["FileName"] = image_path.name
    row["expected_label"] = expected_label_from_filename(image_path)
    row["status"] = "error"
    row["error"] = str(error)
    row["is_correct"] = "False"
    row["match_basis"] = "request failed"
    return row


def write_rows(rows, output_csv):
    if not rows:
        return

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_accuracy_report(rows, report_path, image_dir):
    successful_rows = [row for row in rows if row["status"] == "ok"]
    correct_rows = [row for row in successful_rows if row["is_correct"] == "True"]
    failed_rows = [row for row in rows if row["status"] == "error"]

    total = len(rows)
    successful_count = len(successful_rows)
    correct_count = len(correct_rows)
    accuracy = correct_count / successful_count if successful_count else 0

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write("PlantNet API Accuracy Report\n")
        report_file.write("============================\n\n")
        report_file.write(f"Image folder: {image_dir}\n")
        report_file.write(f"Total images attempted: {total}\n")
        report_file.write(f"Successful API responses: {successful_count}\n")
        report_file.write(f"Failed API responses: {len(failed_rows)}\n")
        report_file.write(f"Correct top-result matches: {correct_count}\n")
        report_file.write(f"Accuracy on successful responses: {accuracy:.3f}\n\n")

        report_file.write("Scoring rule\n")
        report_file.write("------------\n")
        report_file.write(
            "Expected labels are inferred from filenames. A result is counted as correct "
            "when every word in the filename label appears in PlantNet's top scientific "
            "name, best match, or common names.\n\n"
        )

        report_file.write("Per-image results\n")
        report_file.write("-----------------\n")
        for row in rows:
            report_file.write(
                f"{row['FileName']}: expected '{row['expected_label']}', "
                f"PlantNet top result '{row['top_full_scientific_name'] or row['bestMatch']}', "
                f"correct={row['is_correct']} ({row['match_basis']})\n"
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Identify test flower photos with Pl@ntNet.")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--project", default=os.environ.get("PLANTNET_PROJECT", DEFAULT_PROJECT))
    parser.add_argument("--organ", default=os.environ.get("PLANTNET_ORGAN", DEFAULT_ORGAN))
    parser.add_argument("--lang", default=os.environ.get("PLANTNET_LANG", DEFAULT_LANGUAGE))
    parser.add_argument("--nb-results", type=int, default=DEFAULT_NB_RESULTS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--delay", type=float, default=DEFAULT_REQUEST_DELAY_SECONDS)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()
    if args.results_dir != DEFAULT_RESULTS_DIR:
        args.output_csv = args.results_dir / "plantnet_identifications.csv"
        args.report_path = args.results_dir / "plantnet_accuracy_report.txt"
        args.raw_dir = args.results_dir / "raw_json"

    images = find_images(args.image_dir)
    if args.limit is not None:
        images = images[: args.limit]

    if not images:
        raise SystemExit(f"No JPG or PNG images found in {args.image_dir}")

    if args.dry_run:
        print(f"Found {len(images)} image(s) in {args.image_dir}:")
        for image_path in images:
            print(f"- {image_path.name}")
        return

    api_key = os.environ.get("PLANTNET_API_KEY")
    if not api_key:
        raise SystemExit("Missing PLANTNET_API_KEY. Add it to .env or export it in your shell.")

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for index, image_path in enumerate(images, start=1):
        print(f"[{index}/{len(images)}] Identifying {image_path.name}")
        try:
            result = identify_image(
                image_path=image_path,
                api_key=api_key,
                project=args.project,
                organ=args.organ,
                language=args.lang,
                nb_results=args.nb_results,
            )
        except PlantNetRequestError as error:
            print(f"PlantNet request failed for {image_path.name}: {error}")
            rows.append(summarize_error(image_path, error))
            continue

        raw_output_path = args.raw_dir / f"{image_path.stem}.json"
        with open(raw_output_path, "w", encoding="utf-8") as raw_file:
            json.dump(result, raw_file, indent=2)

        row = summarize_result(image_path, result)
        row["raw_json"] = str(raw_output_path.relative_to(PROJECT_ROOT))
        is_correct, match_basis = score_prediction(row["expected_label"], row)
        row["is_correct"] = str(is_correct)
        row["match_basis"] = match_basis
        rows.append(row)

        if index < len(images):
            time.sleep(args.delay)

    write_rows(rows, args.output_csv)
    write_accuracy_report(rows, args.report_path, args.image_dir)
    print(f"Saved {len(rows)} identifications to {args.output_csv}")
    print(f"Saved accuracy report to {args.report_path}")


if __name__ == "__main__":
    main()
