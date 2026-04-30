"""
Convert JPG/JPEG images to PNG in-place while preserving filenames and folders.
Optionally delete original JPG/JPEG files.
"""
from pathlib import Path
import cv2

INPUT_FOLDERS = [
    "data/train_data/clean_images",
    "data/train_data/stego_images",
    "data/val_data/clean_images",
    "data/val_data/stego_images",
    "data/test_data/clean_images",
    "data/test_data/stego_images",
]

# Set to True to delete original JPG/JPEG files after conversion
DELETE_ORIGINALS = False


def convert_folder(folder_path: Path) -> None:
    if not folder_path.exists():
        print(f"Skipping missing folder: {folder_path}")
        return

    jpg_files = list(folder_path.glob("*.[jJ][pP][gG]")) + list(folder_path.glob("*.[jJ][pP][eE][gG]"))
    if not jpg_files:
        print(f"No JPG files found in {folder_path}")
        return

    for jpg_path in jpg_files:
        img = cv2.imread(str(jpg_path))
        if img is None:
            print(f"Failed to read: {jpg_path}")
            continue

        png_path = jpg_path.with_suffix(".png")
        success = cv2.imwrite(str(png_path), img)
        if not success:
            print(f"Failed to write: {png_path}")
            continue

        if DELETE_ORIGINALS:
            try:
                jpg_path.unlink()
            except OSError as exc:
                print(f"Failed to delete {jpg_path}: {exc}")

    print(f"Converted {len(jpg_files)} files in {folder_path}")


def main() -> None:
    for folder in INPUT_FOLDERS:
        convert_folder(Path(folder))


if __name__ == "__main__":
    main()
