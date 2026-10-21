from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

from recsys_lab.config import Settings


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "recsys-lab/0.1"})
    with urlopen(request, timeout=60) as response, temp_path.open("wb") as out:
        shutil.copyfileobj(response, out)
    temp_path.replace(destination)


def _local_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_movielens(settings: Settings, force: bool = False) -> Path:
    archive = settings.paths.raw / settings.data.archive_name
    extracted = settings.paths.raw / settings.data.extracted_dir

    if extracted.exists() and not force:
        return extracted

    if force:
        if archive.exists():
            archive.unlink()
        if extracted.exists():
            shutil.rmtree(extracted)

    if not archive.exists():
        print(f"Downloading {settings.data.dataset} ...")
        _download(settings.data.url, archive)

    expected = settings.data.expected_md5.lower()
    actual = _local_md5(archive)
    if actual != expected:
        archive.unlink(missing_ok=True)
        raise RuntimeError(
            f"MD5 mismatch for {archive.name}: expected {expected}, got {actual}. "
            "Deleted the corrupted archive; rerun the command."
        )

    with zipfile.ZipFile(archive) as zf:
        zf.extractall(settings.paths.raw)

    if not extracted.exists():
        raise RuntimeError(f"Expected extracted directory not found: {extracted}")
    print(f"Dataset ready: {extracted}")
    return extracted
