"""Helpers for pulling the QR dataset from Google Drive and unpacking it into data/raw."""

import shutil
import tempfile
import zipfile
from pathlib import Path

import gdown

DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1i5dQyllOAbZAM4XvXrJ0d1MuUuJMGVeb?usp=sharing"

# Written after a full download and extraction so a partial run is never mistaken for a finished one
EXTRACT_MARKER = ".extracted"
DOWNLOAD_MARKER = ".download_complete"


def extract_zip(zip_path: Path, dest: Path) -> Path:
    """Extract a zip into dest, skipping files that already exist with the same size.

    The folders inside the zip (QR_All_benign, QR_All_Malicious) land directly in dest, so the
    zip's own name never appears in the path. Only if an archive has loose files at its root is
    it nested under a folder named after the zip.
    """
    zip_path, dest = Path(zip_path), Path(dest)
    with zipfile.ZipFile(zip_path) as zf:
        infos = [i for i in zf.infolist() if not i.filename.startswith("__MACOSX")]
        tops = sorted({i.filename.split("/")[0] for i in infos if i.filename.strip("/")})
        loose_files = any(not i.is_dir() and "/" not in i.filename for i in infos)
        target = dest / zip_path.stem if loose_files else dest
        target.mkdir(parents=True, exist_ok=True)

        new, skipped = 0, 0
        for info in infos:
            out = target / info.filename
            if info.is_dir():
                out.mkdir(parents=True, exist_ok=True)
            elif out.exists() and out.stat().st_size == info.file_size:
                skipped += 1
            else:
                zf.extract(info, target)
                new += 1
    print(f"{zip_path.name}: top-level {tops}, {new:,} files extracted, {skipped:,} already present")
    return target


def _ensure_zips(zip_dir: Path, url: str) -> list:
    """Return the valid zips in zip_dir, downloading them from Drive first if needed."""
    zip_dir.mkdir(parents=True, exist_ok=True)
    marker = zip_dir / DOWNLOAD_MARKER
    if not marker.exists():
        # A download cut short leaves a truncated zip behind, so clear those before resuming
        for z in zip_dir.rglob("*.zip"):
            if not zipfile.is_zipfile(z):
                z.unlink()
        gdown.download_folder(url=url, output=str(zip_dir), quiet=False, use_cookies=False, resume=True)
        marker.touch()
    zips = sorted(zip_dir.rglob("*.zip"))
    if not zips:
        marker.unlink(missing_ok=True)
        raise FileNotFoundError("No zip files were downloaded. Check the Drive folder sharing settings.")
    return zips


def load_raw_data(raw_dir, url: str = DRIVE_FOLDER_URL, cache_dir=None) -> Path:
    """Download the dataset zips and unzip them into raw_dir. Safe to call repeatedly.

    cache_dir: optional folder that keeps the zips between runs (on Colab, a folder on Google
    Drive). The zips are then downloaded once, and every later run copies them to fast local
    disk before unzipping, because reading about a million small files straight from a mounted
    Drive is very slow. Without cache_dir the zips go to a temporary folder and are discarded.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    marker = raw_dir / EXTRACT_MARKER
    if marker.exists():
        print(f"data/raw already loaded, skipping download: {raw_dir}")
        return raw_dir

    with tempfile.TemporaryDirectory() as tmp:
        zips = _ensure_zips(Path(cache_dir) if cache_dir else Path(tmp) / "zips", url)
        for z in zips:
            if cache_dir:
                local_zip = Path(tmp) / z.name
                print(f"Copying {z.name} from cache to local disk...")
                shutil.copy2(z, local_zip)
            else:
                local_zip = z
            extract_zip(local_zip, raw_dir)
            if cache_dir:
                local_zip.unlink()

    marker.touch()
    return raw_dir
