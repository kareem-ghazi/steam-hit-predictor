#!/usr/bin/env python3
"""
Steam Hit Predictor - Kaggle Dataset Uploader
Uploads the local cleaned_data/ dataset splits to Kaggle Datasets and reports
the exact Kaggle web URL and in-kernel input path (/kaggle/input/<dataset-slug>).
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Tuple, Optional
from dotenv import load_dotenv

# Search and load .env from root or current directory
PROJECT_ROOT = Path(__file__).resolve().parent
for env_path in [PROJECT_ROOT / ".env", PROJECT_ROOT / "llm-pipeline" / ".env", Path(".env")]:
    if env_path.exists():
        load_dotenv(env_path)
        break


def setup_kaggle_auth() -> Tuple[bool, Optional[str]]:
    """
    Sets up Kaggle authentication using the modern API Token (KAGGLE_API_TOKEN) from .env.
    Returns (is_authenticated, username).
    """
    api_token = os.getenv("KAGGLE_API_TOKEN", "").strip()
    if not api_token or api_token == "your_kaggle_api_token":
        return False, None

    os.environ["KAGGLE_API_TOKEN"] = api_token

    # Write ~/.kaggle/access_token so both kagglehub and kaggle CLI recognize the token
    try:
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        token_file = kaggle_dir / "access_token"
        token_file.write_text(api_token, encoding="utf-8")
    except Exception as e:
        print(f"Notice: Could not write ~/.kaggle/access_token: {e}")

    try:
        import kagglehub
        from kagglehub.config import set_kaggle_api_token
        set_kaggle_api_token(api_token)
        user_info = kagglehub.whoami(verbose=False)
        username = user_info.get("username")
        return True, username
    except Exception as e:
        print(f"Kaggle authentication check: {e}")
        return True, None


def resolve_local_dataset_dir(data_dir: str = "cleaned_data") -> Path:
    """Finds and validates the local dataset split directory."""
    candidates = [
        Path(data_dir),
        PROJECT_ROOT / data_dir,
        PROJECT_ROOT / "cleaned_data",
        Path("cleaned_data"),
    ]
    for c in candidates:
        if c.exists() and (c / "train" / "games.csv").exists():
            return c.resolve()

    # Search recursively for train/games.csv
    for c in candidates:
        if c.exists():
            for p in c.glob("**/train/games.csv"):
                return p.parent.parent.resolve()

    raise FileNotFoundError(
        f"Could not find valid cleaned dataset splits in '{data_dir}'. "
        f"Expected subdirectories 'train/', 'valid/', 'test/' containing 'games.csv'."
    )


def upload_dataset_to_kaggle(
    data_dir: str = "cleaned_data",
    slug: str = "steam-hit-predictor-cleaned-data",
    title: str = "Steam Hit Predictor Cleaned Data",
    version_notes: str = "Cleaned Steam dataset splits for QLoRA training",
    is_public: bool = False
) -> dict:
    """
    Uploads the dataset to Kaggle and returns a dictionary with URLs and in-kernel paths.
    """
    is_auth, username = setup_kaggle_auth()
    if not is_auth or not username:
        print("\n" + "=" * 70)
        print(" [!] Kaggle Authentication Required")
        print("=" * 70)
        print("Please configure your modern Kaggle API token in .env:")
        print("1. Go to https://www.kaggle.com/settings/api")
        print("2. Under 'API Tokens', click 'Generate New Token'")
        print("3. Copy the token into your .env file as:")
        print("   KAGGLE_API_TOKEN=<your_copied_token>")
        print("=" * 70 + "\n")
        sys.exit(1)

    local_path = resolve_local_dataset_dir(data_dir)
    handle = f"{username}/{slug}"
    kaggle_url = f"https://www.kaggle.com/datasets/{handle}"
    kernel_input_path = f"/kaggle/input/{slug}"

    print("==================================================================")
    print("                KAGGLE DATASET UPLOADER                          ")
    print("==================================================================")
    print(f"Kaggle User:        {username}")
    print(f"Dataset Slug:       {slug}")
    print(f"Dataset Handle:     {handle}")
    print(f"Local Source Path:  {local_path}")
    print(f"Version Notes:      {version_notes}")
    print("------------------------------------------------------------------")

    # List splits and file sizes
    files_to_upload = list(local_path.glob("**/*"))
    data_files = [f for f in files_to_upload if f.is_file() and not f.name.startswith(".git")]
    total_bytes = sum(f.stat().st_size for f in data_files)
    total_mb = total_bytes / (1024 * 1024)
    print(f"Files to upload:    {len(data_files)} files ({total_mb:.1f} MB total)")
    for f in data_files:
        rel = f.relative_to(local_path)
        print(f"  - {str(rel):<35} ({f.stat().st_size / (1024*1024):>6.2f} MB)")
    print("==================================================================\n")

    # Check if dataset already exists via Kaggle API
    dataset_exists = False
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        status = str(api.dataset_status(handle))
        print(f"Dataset '{handle}' already exists on Kaggle (status: {status}). Uploading new version...")
        dataset_exists = True
    except Exception:
        print(f"Dataset '{handle}' does not exist yet. Creating new Kaggle dataset...")
        dataset_exists = False

    # Execute upload via kagglehub
    try:
        import kagglehub
        print(f"Uploading files from '{local_path}' to Kaggle ({handle})...")
        kagglehub.dataset_upload(
            handle=handle,
            local_dataset_dir=str(local_path),
            version_notes=version_notes
        )
        print("Dataset upload finished successfully!")
    except Exception as e:
        print(f"Notice from kagglehub upload: {e}")
        # Fallback to kaggle CLI / API if kagglehub raised an exception
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            if dataset_exists:
                api.dataset_create_version(
                    folder=str(local_path),
                    version_notes=version_notes,
                    delete_old_versions=False
                )
            else:
                meta = {
                    "title": title,
                    "id": handle,
                    "licenses": [{"name": "CC0-1.0"}],
                    "isPrivate": not is_public
                }
                meta_path = local_path / "dataset-metadata.json"
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
                api.dataset_create_new(
                    folder=str(local_path),
                    public=is_public,
                    quiet=False
                )
                if meta_path.exists():
                    meta_path.unlink()
        except Exception as e2:
            print(f"Upload error: {e2}")
            raise e2

    result = {
        "dataset_handle": handle,
        "kaggle_web_url": kaggle_url,
        "kaggle_input_path": kernel_input_path,
        "kaggle_splits": {
            "train_games": f"{kernel_input_path}/train/games.csv",
            "train_descriptions": f"{kernel_input_path}/train/games_descriptions.csv",
            "valid_games": f"{kernel_input_path}/valid/games.csv",
            "valid_descriptions": f"{kernel_input_path}/valid/games_descriptions.csv",
            "test_games": f"{kernel_input_path}/test/games.csv",
            "test_descriptions": f"{kernel_input_path}/test/games_descriptions.csv"
        }
    }

    print("\n" + "=" * 70)
    print("                 KAGGLE DATASET UPLOAD COMPLETE                   ")
    print("==================================================================")
    print(f" Kaggle Web URL:      {result['kaggle_web_url']}")
    print(f" Kaggle Dataset:      {result['dataset_handle']}")
    print(f" In-Kernel Path:      {result['kaggle_input_path']}")
    print("------------------------------------------------------------------")
    print(" To use this dataset on Kaggle:")
    print(f"   python llm-pipeline/qlora_training.py --data-dir {result['kaggle_input_path']}")
    print(" Or use the dataset handle directly:")
    print(f"   python llm-pipeline/qlora_training.py --data-dir {result['dataset_handle']}")
    print("=" * 70 + "\n")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Upload cleaned Steam dataset to Kaggle Datasets and report in-kernel path"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="cleaned_data",
        help="Local directory containing train/valid/test splits (default: cleaned_data)"
    )
    parser.add_argument(
        "--slug",
        type=str,
        default="steam-hit-predictor-cleaned-data",
        help="Kaggle dataset slug (default: steam-hit-predictor-cleaned-data)"
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Steam Hit Predictor Cleaned Data",
        help="Kaggle dataset title"
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="Cleaned Steam dataset splits for QLoRA training",
        help="Version notes for the dataset upload"
    )
    parser.add_argument(
        "--public",
        action="store_true",
        default=False,
        help="Make dataset public (default: private)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Print JSON output format"
    )

    args = parser.parse_args()

    res = upload_dataset_to_kaggle(
        data_dir=args.data_dir,
        slug=args.slug,
        title=args.title,
        version_notes=args.notes,
        is_public=args.public
    )

    if args.json:
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
