"""Download helpers for private MEG data files used in PyMEGDec CI/workflows."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess  # nosec B404
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


def _require_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise SystemExit(f"{name} is empty")
    return value


def _parse_file_indices(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    indices: list[int] = []
    for token in re.split(r"[\s,]+", value):
        if not token:
            continue
        try:
            index = int(token)
        except ValueError as exc:
            raise ValueError(f"File indices must be positive integers: {value!r}") from exc
        if index < 1:
            raise ValueError(f"File indices are 1-based and must be positive: {value!r}")
        indices.append(index)
    return indices


def _parse_file_names(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    names = [token.strip() for token in re.split(r"[\s,]+", value) if token.strip()]
    for name in names:
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"File names must be relative paths below the WebDAV root: {name!r}")
    return names


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
    parser = argparse.ArgumentParser(prog=prog, description="Download private MEG data files for local runs and CI workflows.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--source",
        choices=("url-list", "webdav-rclone"),
        default="url-list",
        help="Download source. url-list preserves the historical MEG_DATA_URL_LIST behavior; webdav-rclone uses OwnCloud/WebDAV via rclone.",
    )
    parser.add_argument("--env-name", default="MEG_DATA_URL_LIST")
    parser.add_argument("--webdav-url-env", default="BUSHMEG_WEBDAV_URL")
    parser.add_argument("--webdav-user-env", default="BUSHMEG_DATA_KEY")
    parser.add_argument("--webdav-password-env", default="BUSHMEG_DATA_PASSWORD")
    parser.add_argument("--remote-path", default="", help="Optional path below the WebDAV share root.")
    parser.add_argument("--file-indices", default=None, help="Optional 1-based indices into the remote file list, e.g. '2,3'.")
    parser.add_argument("--file-names", default=None, help="Optional remote file names or relative paths to download, e.g. 'Part2CueData.mat,Part2Data.mat'.")
    parser.add_argument("--rclone-binary", default="rclone")
    parser.add_argument("--rclone-list-timeout-s", type=int, default=300, help="Maximum seconds allowed for the rclone WebDAV listing call.")
    parser.add_argument("--rclone-copy-timeout-s", type=int, default=1800, help="Maximum seconds allowed for each rclone file copy.")
    parser.add_argument("--manifest", default="data-manifest/downloaded-files.txt")
    return parser


def _webdav_remote(path: str = "") -> str:
    clean = path.strip("/")
    return f":webdav:{clean}" if clean else ":webdav:"


def _rclone_options(*, webdav_url: str, webdav_user: str, obscured_password: str) -> list[str]:
    return [
        "--webdav-url",
        webdav_url,
        "--webdav-vendor",
        "owncloud",
        "--webdav-user",
        webdav_user,
        "--webdav-pass",
        obscured_password,
        "--contimeout",
        "30s",
        "--timeout",
        "2m",
        "--retries",
        "3",
        "--low-level-retries",
        "3",
    ]


def _run_rclone(args: list[str], *, capture_output: bool = False, timeout: int | None = None, description: str = "rclone") -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(args, check=False, text=True, capture_output=capture_output, timeout=timeout)  # nosec B603
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(f"{description} timed out after {timeout} seconds.") from exc
    if result.returncode != 0:
        detail = f": {result.stderr.strip()}" if capture_output and result.stderr else ""
        raise SystemExit(f"{description} failed with exit code {result.returncode}{detail}")
    return result


def _download_from_url_list(args: argparse.Namespace, data_dir: Path) -> list[Path]:
    urls = _urls_from_env(args.env_name)
    if not urls:
        raise SystemExit(f"{args.env_name} is empty")

    _prepare_data_dir(data_dir)
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
    return downloaded


def _prepare_data_dir(data_dir: Path) -> None:
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)


def _select_named_remote_files(remote_files: list[str], selected_names: list[str]) -> list[str]:
    selected: list[str] = []
    for name in selected_names:
        matches = [remote_file for remote_file in remote_files if name in {remote_file, Path(remote_file).name}]
        if not matches:
            raise SystemExit(f"Requested WebDAV file {name!r} was not found.")
        if len(matches) > 1:
            raise SystemExit(f"Requested WebDAV file name {name!r} is ambiguous; pass a relative path instead.")
        selected.append(matches[0])
    return selected


def _download_from_webdav_rclone(args: argparse.Namespace, data_dir: Path) -> list[Path]:
    webdav_url = _require_env(args.webdav_url_env)
    webdav_user = _require_env(args.webdav_user_env)
    webdav_password = _require_env(args.webdav_password_env)
    try:
        selected_indices = _parse_file_indices(args.file_indices)
        selected_names = _parse_file_names(args.file_names)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if selected_indices is not None and selected_names is not None:
        raise SystemExit("Pass either --file-indices or --file-names, not both.")

    obscure_result = _run_rclone([args.rclone_binary, "obscure", webdav_password], capture_output=True, timeout=60, description="rclone obscure")
    obscured_password = obscure_result.stdout.strip()
    rclone_options = _rclone_options(webdav_url=webdav_url, webdav_user=webdav_user, obscured_password=obscured_password)

    print(f"Listing WebDAV files below {args.remote_path!r}...", flush=True)
    list_result = _run_rclone(
        [
            args.rclone_binary,
            "lsf",
            _webdav_remote(args.remote_path),
            "--files-only",
            "--recursive",
            "--format",
            "p",
            *rclone_options,
        ],
        capture_output=True,
        timeout=args.rclone_list_timeout_s,
        description="rclone WebDAV listing",
    )
    remote_files = [line.strip() for line in list_result.stdout.splitlines() if line.strip()]
    if not remote_files:
        raise SystemExit(f"No files found in WebDAV remote path {args.remote_path!r}.")
    if selected_indices is not None:
        missing = [index for index in selected_indices if index > len(remote_files)]
        if missing:
            raise SystemExit(f"File indices {missing!r} exceed the {len(remote_files)} files available in the WebDAV listing.")
        remote_files = [remote_files[index - 1] for index in selected_indices]
    elif selected_names is not None:
        remote_files = _select_named_remote_files(remote_files, selected_names)

    _prepare_data_dir(data_dir)
    downloaded: list[Path] = []
    print(f"Downloading {len(remote_files)} WebDAV file(s) to {data_dir}...", flush=True)
    for index, remote_file in enumerate(remote_files, start=1):
        remote_source = _webdav_remote("/".join(part for part in [args.remote_path.strip("/"), remote_file] if part))
        target = data_dir / Path(remote_file).name
        print(f"Downloading file {index}/{len(remote_files)}: {remote_file} -> {target.name}", flush=True)
        _run_rclone(
            [
                args.rclone_binary,
                "copyto",
                remote_source,
                str(target),
                "--progress",
                "--stats",
                "30s",
                *rclone_options,
            ],
            timeout=args.rclone_copy_timeout_s,
            description=f"rclone copy {remote_file!r}",
        )
        downloaded.append(target)
        print(f"Downloaded file {index}/{len(remote_files)}: {target.name}", flush=True)
    return downloaded


def download_meg_data_files(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    parser = _build_parser(prog=prog)
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)

    if args.source == "webdav-rclone":
        downloaded = _download_from_webdav_rclone(args, data_dir)
    else:
        downloaded = _download_from_url_list(args, data_dir)

    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(str(path) for path in downloaded) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(download_meg_data_files())
