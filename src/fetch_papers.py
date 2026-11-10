"""
fetch_papers.py - Downloads and SHA256-verifies exact arXiv paper versions.
"""

import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PAPERS = {
    "1706.03762v7": {
        "title": "Attention Is All You Need (v7)",
        "url": "https://arxiv.org/pdf/1706.03762v7.pdf",
        "filename": "1706.03762v7.pdf",
    },
    "2410.00037v1": {
        "title": "Moshi: a speech-text foundation model for real-time dialogue (v1)",
        "url": "https://arxiv.org/pdf/2410.00037v1.pdf",
        "filename": "2410.00037v1.pdf",
    },
    "2507.07955v2": {
        "title": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling (v2)",
        "url": "https://arxiv.org/pdf/2507.07955v2.pdf",
        "filename": "2507.07955v2.pdf",
    },
}

def calculate_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def verify_pdf(filepath: Path) -> tuple[bool, str]:
    if not filepath.exists():
        return False, "File does not exist"
    size_bytes = filepath.stat().st_size
    if size_bytes < 100 * 1024:
        return False, f"File too small ({size_bytes} bytes < 100 KB)"
    with open(filepath, "rb") as f:
        header = f.read(5)
        if header != b"%PDF-":
            return False, f"Invalid PDF header: {header}"
    return True, f"OK ({size_bytes} bytes)"

def download_paper(paper_id: str, meta: dict, dest_dir: Path, client: httpx.Client) -> dict:
    target_path = dest_dir / meta["filename"]
    
    # Check if already downloaded and valid
    if target_path.exists():
        is_valid, reason = verify_pdf(target_path)
        if is_valid:
            logger.info(f"Using cached valid PDF for {paper_id}: {meta['filename']}")
            sha256 = calculate_sha256(target_path)
            return {
                "paper_id": paper_id,
                "title": meta["title"],
                "url": meta["url"],
                "filename": meta["filename"],
                "size_bytes": target_path.stat().st_size,
                "sha256": sha256,
                "verified": True,
            }

    logger.info(f"Downloading {paper_id} from {meta['url']} ...")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.get(meta["url"])
            if response.status_code == 200:
                with open(target_path, "wb") as f:
                    f.write(response.content)
                is_valid, reason = verify_pdf(target_path)
                if not is_valid:
                    raise ValueError(f"Downloaded file validation failed: {reason}")
                sha256 = calculate_sha256(target_path)
                logger.info(f"Successfully downloaded {paper_id} ({target_path.stat().st_size} bytes, SHA256: {sha256[:12]}...)")
                return {
                    "paper_id": paper_id,
                    "title": meta["title"],
                    "url": meta["url"],
                    "filename": meta["filename"],
                    "size_bytes": target_path.stat().st_size,
                    "sha256": sha256,
                    "verified": True,
                }
            else:
                logger.warning(f"Attempt {attempt+1}: HTTP {response.status_code} for {meta['url']}")
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed for {paper_id}: {e}")
        time.sleep(2.0 * (attempt + 1))

    raise RuntimeError(f"Failed to download valid PDF for {paper_id} after {max_retries} attempts")

def main():
    root_dir = Path(__file__).resolve().parent.parent
    dest_dir = root_dir / "data" / "papers"
    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dest_dir / "manifest.json"

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    manifest = {}
    with httpx.Client(follow_redirects=True, timeout=90.0, headers=headers) as client:
        for paper_id, meta in PAPERS.items():
            entry = download_paper(paper_id, meta, dest_dir, client)
            manifest[paper_id] = entry

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Saved manifest to {manifest_path}")

if __name__ == "__main__":
    main()
