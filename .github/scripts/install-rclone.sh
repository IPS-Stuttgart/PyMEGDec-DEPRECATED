#!/usr/bin/env bash
set -euo pipefail

if ! command -v rclone >/dev/null 2>&1; then
	mkdir -p "$RUNNER_TEMP/rclone-bin" "$RUNNER_TEMP/rclone-download"
	curl -fsSL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o "$RUNNER_TEMP/rclone-download/rclone.zip"
	python3 - <<'PY'
import os
import shutil
import zipfile
from pathlib import Path

temp = Path(os.environ["RUNNER_TEMP"])
zip_path = temp / "rclone-download" / "rclone.zip"
extract_dir = temp / "rclone-download" / "extract"
bin_dir = temp / "rclone-bin"
with zipfile.ZipFile(zip_path) as archive:
    archive.extractall(extract_dir)
matches = list(extract_dir.glob("rclone-*/rclone"))
if not matches:
    raise SystemExit("Downloaded rclone archive did not contain an rclone binary.")
target = bin_dir / "rclone"
shutil.copy2(matches[0], target)
target.chmod(0o755)
PY
	echo "$RUNNER_TEMP/rclone-bin" >>"$GITHUB_PATH"
	export PATH="$RUNNER_TEMP/rclone-bin:$PATH"
fi

rclone version
