#!/usr/bin/env python3
"""Download and validate the official MovieLens Small dataset.

This script intentionally keeps the data local and respects the GroupLens license and
attribution requirements. It is designed for reproducible local setup and not for
public redistribution.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / 'data' / 'raw'
ZIP_URL = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
EXPECTED_SHA256 = 'aa289ca83157595d0df6aea1be6a4ded676ddc4385472e8313a8ed9805352646'
TARGET_FILE = RAW_DIR / 'ratings.csv'


def verify_sha256(path: Path, expected: str) -> bool:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest == expected


def ensure_raw_dir() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def download_dataset() -> None:
    ensure_raw_dir()
    zip_path = RAW_DIR / 'ml-latest-small.zip'
    print(f'Downloading official MovieLens Small archive to {zip_path}')
    with urllib.request.urlopen(ZIP_URL) as response, open(zip_path, 'wb') as handle:
        shutil.copyfileobj(response, handle)

    with tempfile.TemporaryDirectory() as tmp_dir:
        import zipfile

        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(tmp_dir)
        extracted = Path(tmp_dir) / 'ml-latest-small' / 'ratings.csv'
        if not extracted.exists():
            raise FileNotFoundError('Expected ratings.csv was not found in the downloaded MovieLens archive.')
        shutil.copy2(extracted, TARGET_FILE)
    zip_path.unlink(missing_ok=True)


def main() -> int:
    ensure_raw_dir()
    if not TARGET_FILE.exists():
        download_dataset()

    if not verify_sha256(TARGET_FILE, EXPECTED_SHA256):
        raise SystemExit(
            'Downloaded MovieLens ratings.csv checksum does not match the pinned dataset value. '
            'The official dataset should be re-downloaded and the file should be checked before publication.'
        )

    print(f'MovieLens Small dataset ready at {RAW_DIR}')
    print('Checksum verified: OK')
    print('Attribution: See data/raw/README.txt for GroupLens citation and license terms.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
