import os
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import requests

from jaxtari.paths import get_base_detector_dir

# 1. Configuration
DETECTORS_URL = os.environ.get(
    "JAXTARI_DETECTORS_URL",
    "https://drive.google.com/uc?export=download&id=1B4hCG5XD3ZGv0rZd4-RT3CTWeqwYJJqn",
)

# Install where the detection wrapper looks checkpoints up (next to the sprites).
DETECTOR_DIR = Path(get_base_detector_dir())


def _download_archive(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest``. Large Google Drive files answer with a "can't scan for viruses"
    page first; its download form is submitted to get the actual file."""
    session = requests.Session()
    response = session.get(url, stream=True, timeout=60)
    response.raise_for_status()
    if "text/html" in response.headers.get("content-type", ""):
        page = response.text
        form = re.search(r'<form[^>]*id="download-form"[^>]*action="([^"]+)"', page)
        if form is None:
            raise RuntimeError("Unexpected HTML response instead of the detector archive.")
        params = dict(re.findall(r'<input type="hidden" name="([^"]+)" value="([^"]*)"', page))
        response = session.get(form.group(1), params=params, stream=True, timeout=60)
        response.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=1 << 20):
            f.write(chunk)


def _extract_variants(archive_path: Path, dest_dir: Path) -> list[str]:
    """Copy every directory containing .npz checkpoints (e.g. ``gray96/``) to ``dest_dir/<name>``."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(tmp_path)

        variant_dirs = sorted({p.parent for p in tmp_path.rglob("*.npz")})
        if not variant_dirs:
            raise RuntimeError("Invalid archive: no .npz detector checkpoints found.")

        for src in variant_dirs:
            shutil.copytree(src, dest_dir / src.name, dirs_exist_ok=True)
        return [p.name for p in variant_dirs]


def download_and_extract() -> None:
    """Download and install the detector checkpoints (used by jaxtari.detection)."""
    print(f"Downloading detectors from {DETECTORS_URL}...")
    try:
        DETECTOR_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp_dir:
            archive_path = Path(tmp_dir) / "detectors.zip"
            _download_archive(DETECTORS_URL, archive_path)
            print("Extracting detectors...")
            variants = _extract_variants(archive_path, DETECTOR_DIR)

        counts = ", ".join(f"{v} ({len(list((DETECTOR_DIR / v).glob('*.npz')))} games)" for v in variants)
        print(f"✅ Success! Detectors installed to: {DETECTOR_DIR} [{counts}]")

    except Exception as e:
        print(f"❌ Error downloading/installing detectors: {e}")
        sys.exit(1)


if __name__ == "__main__":
    download_and_extract()
