import csv
import os
from pathlib import Path

# Paths
CSV_PATH = "captions/dataset_captionsV2.csv"
IMAGES_DIR = "images"

def find_actual_extension(stem: str, images_dir: str) -> str | None:
    """Find the actual extension of a file stem by checking what exists on disk."""
    for ext in [".png", ".jpg", ".jpeg"]:
        full_name = f"{stem}{ext}"
        if os.path.exists(os.path.join(images_dir, full_name)):
            return ext
    return None  # File not found at all

def main():
    rows_to_update = []

    with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        all_rows = list(reader)

    for row in all_rows:
        changed = False
        for col in ["lost_image", "found_image"]:
            current_val = row[col]
            stem = Path(current_val).stem
            current_ext = Path(current_val).suffix

            actual_ext = find_actual_extension(stem, IMAGES_DIR)
            if actual_ext and actual_ext != current_ext:
                row[col] = f"{stem}{actual_ext}"
                changed = True
                print(f"  UPDATE: {current_val} -> {row[col]}")
            elif not actual_ext:
                print(f"  MISSING: {current_val} (no matching file found)")

        if changed:
            rows_to_update.append(row)

    # Write back
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nDone. Updated {len(rows_to_update)} row(s) in {CSV_PATH}.")

if __name__ == "__main__":
    main()