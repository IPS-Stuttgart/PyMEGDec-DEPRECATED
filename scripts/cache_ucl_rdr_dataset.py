"""Cache the public UCL RDR MEG navigation dataset on a runner.

The dataset's all-in-one article download endpoint can be awkward for command-line
clients, but the public Figshare API exposes stable per-file downloader URLs plus
sizes and MD5 checksums. This script downloads missing files into a persistent
local cache and skips files that are already complete.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess  # nosec B404
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, TypedDict

DEFAULT_ARTICLE_ID = "31277950"
DEFAULT_VERSION = "1"
DEFAULT_CACHE_DIR = "$HOME/.cache/datasets/ucl-rdr-31277950"


def require_https_url(url: str) -> str:
    """Return ``url`` after verifying that it uses HTTPS."""

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"Only HTTPS URLs are supported: {url!r}")
    return url


@dataclass(frozen=True)
class DatasetFile:
    """One downloadable file from the UCL RDR/Figshare metadata."""

    file_id: int
    name: str
    size: int
    download_url: str
    md5: str
    group: str
    participant: int | None


class ManifestRow(TypedDict):
    """One row written to the CSV manifest and JSON summary."""

    name: str
    file_id: int
    group: str
    participant: int | str
    size: int
    size_gb: str
    md5: str
    status: str
    downloaded: str
    path: str
    download_url: str


def parse_participants(spec: str) -> set[int] | None:
    """Parse participant ranges such as ``1-4,6,8``.

    ``all`` and the empty string return ``None`` to indicate no participant
    filtering.
    """

    cleaned = spec.strip().lower()
    if not cleaned or cleaned == "all":
        return None

    participants: set[int] = set()
    for token in cleaned.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, stop_text = token.split("-", maxsplit=1)
            start = int(start_text)
            stop = int(stop_text)
            if stop < start:
                raise ValueError(f"Invalid participant range: {token!r}")
            participants.update(range(start, stop + 1))
        else:
            participants.add(int(token))
    return participants


def parse_file_groups(spec: str) -> set[str]:
    """Parse file group filters.

    Supported groups are:
    ``all``, ``behavioural``, ``cfix``, ``taskkeys``, ``mat``, and ``dat``.
    """

    groups = {item.strip().lower() for item in spec.split(",") if item.strip()}
    if not groups or "all" in groups:
        return {"all"}
    allowed = {"behavioural", "cfix", "taskkeys", "mat", "dat"}
    unknown = groups - allowed
    if unknown:
        raise ValueError(f"Unsupported file group(s): {', '.join(sorted(unknown))}")
    return groups


def fetch_metadata(article_id: str, version: str) -> dict[str, Any]:
    """Fetch public Figshare/UCL RDR metadata."""

    url = require_https_url(f"https://api.figshare.com/v2/articles/{article_id}/versions/{version}")
    # URL is an HTTPS Figshare API endpoint built from fixed path components.
    with urllib.request.urlopen(url) as response:  # nosec B310
        return json.load(response)


def expand_cache_dir(path_text: str) -> Path:
    """Expand ``$HOME``/``~`` cache paths consistently across platforms."""

    if path_text.startswith("$HOME"):
        path_text = str(Path.home()) + path_text[len("$HOME") :]
    return Path(os.path.expandvars(path_text)).expanduser().resolve()


def classify_file(raw_file: dict[str, Any]) -> DatasetFile:
    """Convert a Figshare file metadata dict into a typed file record."""

    name = raw_file["name"]
    participant_match = re.search(r"Part(\d+)", name)
    participant = int(participant_match.group(1)) if participant_match else None
    if name == "BehaviouralData.mat":
        group = "behavioural"
    elif name.startswith("cFixCueffMp"):
        group = "cfix"
    elif name.startswith("taskKeyseffMp"):
        group = "taskkeys"
    else:
        group = "other"
    return DatasetFile(
        file_id=int(raw_file["id"]),
        name=name,
        size=int(raw_file["size"]),
        download_url=require_https_url(str(raw_file["download_url"])),
        md5=str(raw_file.get("computed_md5") or raw_file.get("supplied_md5") or ""),
        group=group,
        participant=participant,
    )


def select_files(files: Iterable[DatasetFile], participants: set[int] | None, groups: set[str], limit: int | None) -> list[DatasetFile]:
    """Select requested files from dataset metadata."""

    selected: list[DatasetFile] = []
    for item in files:
        extension = item.name.rsplit(".", maxsplit=1)[-1].lower()
        group_match = "all" in groups or item.group in groups or extension in groups
        participant_match = item.participant is None or participants is None or item.participant in participants
        if group_match and participant_match:
            selected.append(item)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def md5sum(path: Path) -> str:
    """Compute an MD5 checksum using chunked reads."""

    hasher = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def file_is_complete(path: Path, expected_size: int, expected_md5: str, verify_md5: bool) -> tuple[bool, str]:
    """Return whether a cached file is complete and a human-readable status."""

    if not path.exists():
        return False, "missing"
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        return False, f"size-mismatch:{actual_size}"
    if verify_md5 and expected_md5:
        actual_md5 = md5sum(path)
        if actual_md5 != expected_md5:
            return False, f"md5-mismatch:{actual_md5}"
    return True, "cached"


def run_curl(curl_path: str, item: DatasetFile, destination: Path, retries: int) -> None:
    """Download one file using curl with resume enabled."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > item.size:
        destination.unlink()

    command = [
        curl_path,
        "-L",
        "--fail",
        "--retry",
        str(retries),
        "--retry-delay",
        "5",
        "--connect-timeout",
        "60",
        "--speed-time",
        "120",
        "--speed-limit",
        "1024",
        "-C",
        "-",
        "-o",
        str(destination),
        "--",
        item.download_url,
    ]
    # Command is an argument list and shell=False by default.
    subprocess.run(command, check=True)  # nosec B603


def write_manifest(path: Path, rows: list[ManifestRow]) -> None:
    """Write a compact CSV manifest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "file_id",
        "group",
        "participant",
        "size",
        "size_gb",
        "md5",
        "status",
        "downloaded",
        "path",
        "download_url",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, metadata: dict[str, Any], rows: list[ManifestRow], cache_dir: Path, dry_run: bool) -> None:
    """Write a JSON summary for workflow step summaries and artifacts."""

    path.parent.mkdir(parents=True, exist_ok=True)
    total_size = sum(row["size"] for row in rows)
    downloaded_size = sum(row["size"] for row in rows if row["downloaded"] == "true")
    cached_count = sum(1 for row in rows if row["status"] == "cached")
    summary = {
        "title": metadata.get("title"),
        "doi": metadata.get("doi"),
        "cache_dir": str(cache_dir),
        "dry_run": dry_run,
        "selected_files": len(rows),
        "selected_size_bytes": total_size,
        "selected_size_gb": total_size / 1_000_000_000,
        "downloaded_files": sum(1 for row in rows if row["downloaded"] == "true"),
        "downloaded_size_bytes": downloaded_size,
        "downloaded_size_gb": downloaded_size / 1_000_000_000,
        "cached_files_after_run": cached_count,
        "missing_or_incomplete_after_run": len(rows) - cached_count,
    }
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--article-id", default=DEFAULT_ARTICLE_ID)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--participants", default="all")
    parser.add_argument("--file-groups", default="all")
    parser.add_argument("--manifest-output", default="outputs/ucl_rdr_cache_manifest.csv")
    parser.add_argument("--summary-output", default="outputs/ucl_rdr_cache_summary.json")
    parser.add_argument("--verify-existing", action="store_true", help="MD5-check already cached files before skipping them.")
    parser.add_argument("--no-verify-downloads", action="store_true", help="Do not MD5-check files that were downloaded in this run.")
    parser.add_argument("--dry-run", action="store_true", help="Print and manifest the selected files without downloading.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of selected files, mainly for smoke tests.")
    parser.add_argument("--curl-path", default=shutil.which("curl") or "curl")
    parser.add_argument("--retries", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    args = parse_args(sys.argv[1:] if argv is None else argv)
    cache_dir = expand_cache_dir(args.cache_dir)
    participants = parse_participants(args.participants)
    groups = parse_file_groups(args.file_groups)
    metadata = fetch_metadata(args.article_id, args.version)
    all_files = [classify_file(raw_file) for raw_file in metadata["files"]]
    selected_files = select_files(all_files, participants, groups, args.limit)
    if not selected_files:
        raise RuntimeError("No files matched the requested participant/file-group filters.")

    rows: list[ManifestRow] = []
    verify_downloads = not args.no_verify_downloads
    print(f"Cache directory: {cache_dir}")
    print(f"Selected files: {len(selected_files)}")
    print(f"Selected size: {sum(item.size for item in selected_files) / 1_000_000_000:.3f} GB")

    for item in selected_files:
        destination = cache_dir / item.name
        complete, status = file_is_complete(destination, item.size, item.md5, args.verify_existing)
        downloaded = False
        if complete:
            print(f"cached: {item.name}")
        elif args.dry_run:
            print(f"would download: {item.name} ({item.size / 1_000_000_000:.3f} GB)")
        else:
            print(f"downloading: {item.name} ({item.size / 1_000_000_000:.3f} GB)")
            run_curl(args.curl_path, item, destination, args.retries)
            downloaded = True
            complete, status = file_is_complete(destination, item.size, item.md5, verify_downloads)
            if not complete:
                raise RuntimeError(f"Downloaded file is incomplete: {destination} ({status})")

        rows.append(
            {
                "name": item.name,
                "file_id": item.file_id,
                "group": item.group,
                "participant": item.participant if item.participant is not None else "",
                "size": item.size,
                "size_gb": f"{item.size / 1_000_000_000:.6f}",
                "md5": item.md5,
                "status": status if complete else "dry-run" if args.dry_run else status,
                "downloaded": str(downloaded).lower(),
                "path": str(destination),
                "download_url": item.download_url,
            }
        )

    manifest_output = Path(args.manifest_output)
    summary_output = Path(args.summary_output)
    write_manifest(manifest_output, rows)
    write_summary(summary_output, metadata, rows, cache_dir, args.dry_run)
    print(f"Wrote manifest: {manifest_output}")
    print(f"Wrote summary: {summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
