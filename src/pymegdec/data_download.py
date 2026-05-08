"""Download helpers for private MEG data files used in PyMEGDec CI/workflows."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import urllib.parse
import urllib.request
from collections.abc import Sequence
from pathlib import Path

_ALLOWED_URL_SCHEMES = {"https"}


class _HTTPSOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Redirect handler that preserves the HTTPS-only download invariant."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001, ANN201
        redirect_url = urllib.parse.urljoin(req.full_url, newurl)
        _validate_https_url(redirect_url, description="redirect")
        return super().redirect_request(req, fp, code, msg, headers, redirect_url)


_DOWNLOAD_OPENER = urllib.request.build_opener(_HTTPSOnlyRedirectHandler)


def _validate_https_url(url: str, *, description: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme.lower() not in _ALLOWED_URL_SCHEMES or not parsed.netloc:
        raise ValueError(f"Only absolute HTTPS {description} URLs are supported: {url!r}")
    return url


def _urls_from_env(name: str) -> list[str]:
    raw = os.environ.get(name, "")
    return [token.strip() for token in re.split(r"[\s,]+", raw) if token.strip()]


def _direct_url(url: str) -> str:
    parsed = urllib.parse.urlparse(_validate_https_url(url, description="source"))
    if parsed.path.rstrip("/").endswith("/download"):
        direct_url = url
    elif "/s/" in f"/{parsed.path.strip('/')}/":
        direct_url = url.rstrip("/") + "/download"
    else:
        direct_url = url
    return _validate_https_url(direct_url, description="download")


def _open_https(request: urllib.request.Request, *, timeout: int):
    _validate_https_url(request.full_url, description="download")
    response = _DOWNLOAD_OPENER.open(request, timeout=timeout)
    _validate_https_url(response.geturl(), description="final download")
    return response


def _filename(response, index: int) -> str:  # noqa: ANN001
    header = response.headers.get("Content-Disposition", "")
    match = re.search(r"filename\*\s*=\s*UTF-8''([^;]+)", header, flags=re.IGNORECASE)
    if match:
        return Path(urllib.parse.unquote(match.group(1))).name
    match = re.search(r'filename\s*=\s*"?([^";]+)"?', header, flags=re.IGNORECASE)
    if match:
        return Path(urllib.parse.unquote(match.group(1))).name
    fallback = Path(urllib.parse.urlparse(response.geturl()).path).name
    if fallback and fallback.lower() != "download":
        return fallback
    return f"downloaded_meg_file_{index:04d}.mat"


def _build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Download MEG data files from HTTPS URLs listed in an environment variable.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--env-name", default="MEG_DATA_URL_LIST")
    parser.add_argument("--manifest", default="data-manifest/downloaded-files.txt")
    return parser


def download_meg_data_files(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    parser = _build_parser(prog=prog)
    args = parser.parse_args(argv)

    urls = _urls_from_env(args.env_name)
    if not urls:
        raise SystemExit(f"{args.env_name} is empty")

    data_dir = Path(args.data_dir)
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    downloaded: list[Path] = []
    for index, url in enumerate(urls, start=1):
        request = urllib.request.Request(_direct_url(url), headers={"User-Agent": "PyMEGDec"})
        with _open_https(request, timeout=180) as response:
            target = data_dir / _filename(response, index)
            counter = 2
            while target.exists():
                target = data_dir / f"{target.stem}_{counter}{target.suffix}"
                counter += 1
            with target.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        downloaded.append(target)
        print(f"Downloaded file #{index}: {target.name}")

    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(str(path) for path in downloaded) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(download_meg_data_files())
