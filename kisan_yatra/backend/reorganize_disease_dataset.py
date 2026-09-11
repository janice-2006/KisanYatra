"""
Reorganize raw PlantVillage-style folders (one per crop+disease, e.g.
"Tomato_Early_blight") into the 8 category buckets expected by
train_disease_model.py / DISEASE_REMEDIES in disease_detection.py:

    powdery_mildew, early_blight, late_blight, leaf_spot,
    rust, bacterial_wilt, anthracnose, healthy

Run this ONCE, pointed at your "crop diseases" folder. It COPIES images
(doesn't move/delete originals), so nothing is destroyed if the mapping
needs tweaking. Any source folder it can't confidently classify is skipped
and printed at the end so you can route it manually.

Usage:
    python reorganize_disease_dataset.py
"""

import shutil
from pathlib import Path

# Folder that currently contains the raw per-crop subfolders
# (e.g. Tomato_Early_blight, Potato_healthy, Pepper_bell_Bacterial_spot...)
SOURCE_ROOT = Path(__file__).resolve().parent / "crop diseases"

# Where the 8 category buckets should end up (same folder, new subfolders)
DEST_ROOT = SOURCE_ROOT

# Folders to ignore entirely (nested zip artifacts, non-class folders,
# and the category folders themselves so a second run doesn't re-copy
# already-sorted images into themselves).
IGNORE_FOLDERS = {
    "plantvillage",
    "powdery_mildew", "early_blight", "late_blight", "leaf_spot",
    "rust", "bacterial_wilt", "anthracnose", "healthy",
}

CATEGORIES = [
    "powdery_mildew", "early_blight", "late_blight", "leaf_spot",
    "rust", "bacterial_wilt", "anthracnose", "healthy",
]

# Keyword → category. Checked in order, first match wins, case-insensitive.
# Covers the standard PlantVillage folder names (Tomato_*, Potato_*,
# Pepper_bell_*, Apple_*, Corn_*, Grape_*, etc.)
KEYWORD_MAP = [
    ("healthy", "healthy"),
    ("powdery_mildew", "powdery_mildew"),
    ("late_blight", "late_blight"),
    ("early_blight", "early_blight"),
    ("bacterial_spot", "bacterial_wilt"),
    ("bacterial_wilt", "bacterial_wilt"),
    ("septoria_leaf_spot", "leaf_spot"),
    ("gray_leaf_spot", "leaf_spot"),
    ("leaf_spot", "leaf_spot"),
    ("target_spot", "leaf_spot"),
    ("rust", "rust"),
    ("leaf_mold", "leaf_spot"),
    ("scab", "leaf_spot"),
    ("black_rot", "anthracnose"),
    ("anthracnose", "anthracnose"),
]

# Folders that don't map cleanly onto any of the 8 categories (viruses,
# mites, mosaic, etc.) — listed here explicitly so they're SKIPPED rather
# than mis-bucketed. Add to DISEASE_REMEDIES yourself later if you want a
# 9th category instead of dropping these.
KNOWN_UNMAPPED_KEYWORDS = [
    "mosaic_virus", "yellowleaf__curl_virus", "spider_mites", "haunted",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def classify_folder_name(folder_name: str):
    name_lower = folder_name.lower()

    for bad_kw in KNOWN_UNMAPPED_KEYWORDS:
        if bad_kw in name_lower:
            return None  # explicitly unmapped

    for keyword, category in KEYWORD_MAP:
        if keyword in name_lower:
            return category

    return None  # no match found


def reorganize():
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"Source folder not found: {SOURCE_ROOT}")

    for category in CATEGORIES:
        (DEST_ROOT / category).mkdir(parents=True, exist_ok=True)

    copied_counts = {cat: 0 for cat in CATEGORIES}
    skipped_folders = []

    for entry in sorted(SOURCE_ROOT.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.lower() in IGNORE_FOLDERS:
            continue

        category = classify_folder_name(entry.name)
        if category is None:
            skipped_folders.append(entry.name)
            continue

        dest_dir = DEST_ROOT / category
        n_copied = 0
        for image_path in entry.rglob("*"):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            # Prefix filename with source folder to avoid collisions when
            # multiple crop folders feed the same category.
            dest_name = f"{entry.name}__{image_path.name}"
            shutil.copy2(image_path, dest_dir / dest_name)
            n_copied += 1

        copied_counts[category] += n_copied
        print(f"{entry.name:40} → {category:16} ({n_copied} images)")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for category in CATEGORIES:
        print(f"{category:20}: {copied_counts[category]:5} images")

    if skipped_folders:
        print("\nSkipped (no confident category match):")
        for name in skipped_folders:
            print(f"  - {name}")
        print(
            "\nThese weren't copied anywhere. If you want them included, "
            "either rename them to include one of the keywords in "
            "KEYWORD_MAP, or move their images manually into a category "
            "folder."
        )

    print("\nDone. Review the counts above, then run train_disease_model.py.")


if __name__ == "__main__":
    reorganize()