#!/usr/bin/env python
"""Fetch the UCI Air Quality dataset into ``data/raw``.

The archive is a zip containing both a CSV and an XLSX of the same series. Only
the CSV is kept: the spreadsheet carries the same numbers with the sentinel
already partly formatted away, which is exactly the ambiguity this project is
about.

The dataset is redistributed by UCI under its own terms; this script downloads
it rather than vendoring a copy.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aqf.config import load_settings  # noqa: E402
from aqf.logging_utils import configure_logging, get_logger  # noqa: E402

logger = get_logger(__name__)

EXPECTED_MEMBER = "AirQualityUCI.csv"
OUTPUT_NAME = "AirQuality.csv"


def parse_args() -> argparse.Namespace:
    settings = load_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=settings.dataset_url, help="source archive")
    parser.add_argument("--output-dir", default=f"{settings.data_dir}/raw", help="where to write")
    parser.add_argument("--force", action="store_true", help="overwrite an existing copy")
    parser.add_argument("--timeout", type=float, default=60.0, help="network timeout in seconds")
    return parser.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()

    destination = Path(args.output_dir) / OUTPUT_NAME
    if destination.exists() and not args.force:
        logger.info("already_present", path=str(destination), hint="pass --force to replace it")
        print(f"{destination} already exists; nothing to do (use --force to replace it).")
        return 0

    logger.info("downloading", url=args.url)
    try:
        with urllib.request.urlopen(args.url, timeout=args.timeout) as response:
            payload = response.read()
    except OSError as exc:
        logger.error("download_failed", url=args.url, error=str(exc))
        print(
            f"Could not download {args.url}: {exc}\n"
            "If this machine has no outbound access, fetch the archive elsewhere and unzip "
            f"{EXPECTED_MEMBER} to {destination}.",
            file=sys.stderr,
        )
        return 1

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = archive.namelist()
            member = next((m for m in members if m.endswith(EXPECTED_MEMBER)), None)
            if member is None:
                logger.error("member_missing", expected=EXPECTED_MEMBER, found=members)
                print(
                    f"The archive does not contain {EXPECTED_MEMBER}; it holds {members}.",
                    file=sys.stderr,
                )
                return 1
            destination.write_bytes(archive.read(member))
    except zipfile.BadZipFile:
        # Some mirrors serve the CSV directly rather than zipped.
        destination.write_bytes(payload)

    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    size_mb = destination.stat().st_size / 1e6
    logger.info("saved", path=str(destination), megabytes=round(size_mb, 2), sha256=digest)
    print(f"Wrote {destination} ({size_mb:.2f} MB)\nsha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
