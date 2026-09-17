"""
model_downloader.py - Reliable Remote Artifact Downloader for Streamlit Deployments
TalentProfile AI - Universal Talent Recommendation System
"""

import os
import sys
import urllib.request
from typing import Dict

# ==============================================================================
# GITHUB RELEASE ARTIFACT CONFIGURATION
# ==============================================================================
# Configured repository and release tag for model hosting
GITHUB_REPO_OWNER = "rockyroshanbehera"
GITHUB_REPO_NAME = "talentprofile-recommendation-system"
RELEASE_TAG = "v1.0.0"

# Direct download links for GitHub Release assets:
BASE_RELEASE_URL = f"https://github.com/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/releases/download/{RELEASE_TAG}"

MODEL_DOWNLOAD_URLS: Dict[str, str] = {
    "tfidf_matrix.pkl": f"{BASE_RELEASE_URL}/tfidf_matrix.pkl",
    "talent_profiles.pkl": f"{BASE_RELEASE_URL}/talent_profiles.pkl",
}


def download_file_with_progress(url: str, dest_path: str, filename: str) -> None:
    temp_path = dest_path + ".tmp"
    print(f"Downloading required model artifact '{filename}' from remote release...")
    print(f"   Source: {url}")

    try:
        def reporthook(count, block_size, total_size):
            if total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                downloaded_mb = (count * block_size) / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                sys.stdout.write(f"\r   Progress: {percent}% ({downloaded_mb:.1f} MB / {total_mb:.1f} MB)")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, temp_path, reporthook=reporthook)
        print("\n   Download complete. Moving artifact to destination...")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        os.rename(temp_path, dest_path)
        print(f"Successfully prepared '{filename}'.")
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        err_msg = (
            f"Failed to download model artifact '{filename}' from URL: {url}\n"
            f"Error details: {str(e)}\n"
            "Please ensure that:\n"
            "1. You have created a GitHub Release with tag matching RELEASE_TAG.\n"
            "2. 'tfidf_matrix.pkl' and 'talent_profiles.pkl' are attached to the release assets.\n"
            "3. GITHUB_REPO_OWNER and GITHUB_REPO_NAME in src/model_downloader.py match your repository."
        )
        raise RuntimeError(err_msg)


def ensure_model_files_exist(models_dir: str) -> None:
    os.makedirs(models_dir, exist_ok=True)

    for filename, url in MODEL_DOWNLOAD_URLS.items():
        file_path = os.path.join(models_dir, filename)
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            download_file_with_progress(url, file_path, filename)
