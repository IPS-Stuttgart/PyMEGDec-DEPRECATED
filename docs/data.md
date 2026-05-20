# Data configuration

PyMEGDec intentionally resolves private or machine-specific data paths at
runtime. Do not commit local paths or participant data files.

## Expected files

Participant data files are expected to use these names:

```text
Part2Data.mat
Part2CueData.mat
```

Replace `2` with the participant id. Workflows that transfer between the main
experiment and cue experiment require both files for a participant.

## Resolution order

PyMEGDec resolves the data directory in this order:

1. A command-line `--data-dir` option, or the `data_folder` argument in the
   Python API.
2. The `PYMEGDEC_DATA_DIR` environment variable.
3. A local `.pymegdec-data-dir` file containing one path. The resolver searches
   the current working directory, its parents, and the project root.
4. The current working directory, preserved for backwards compatibility.

The `.pymegdec-data-dir` file is ignored by git. It may contain an absolute path
or a path relative to the file location. Blank lines and lines starting with
`#` are ignored.

## Examples

Pass the directory explicitly:

```bash
pymegdec stimulus-decoding --data-dir /path/to/MEG-Data --participants 2 --output outputs/part2_stimulus_decoding.csv
```

Set an environment variable on macOS/Linux:

```bash
export PYMEGDEC_DATA_DIR=/path/to/MEG-Data
python -m unittest discover -v
```

Set an environment variable on PowerShell:

```powershell
$env:PYMEGDEC_DATA_DIR = "C:\path\to\MEG-Data"
python -m unittest discover -v
```

Use a local config file:

```bash
echo /path/to/MEG-Data > .pymegdec-data-dir
pymegdec transfer --participant 2 --null-window-center nan
```

## Private-data bootstrap script

Private-data transport is repository infrastructure, not part of the PyMEGDec
Python package. Use the standalone helper script when a CI job or local checkout
needs to materialize the private Bush/MEG files:

```bash
python scripts/download_private_meg_data.py --source webdav-rclone --data-dir data --file-names Part2CueData.mat,Part2Data.mat
```

The script also supports the historical URL-list mode:

```bash
python scripts/download_private_meg_data.py --data-dir data --env-name MEG_DATA_URL_LIST
```

The rclone-backed mode reads `BUSHMEG_WEBDAV_URL`, `BUSHMEG_DATA_KEY`, and
`BUSHMEG_DATA_PASSWORD` from the environment. Without `--file-indices` or
`--file-names`, it downloads all files found recursively in the selected remote
path.

## Participant ranges

Commands that accept multiple participants use compact participant specs such as:

```text
2
1-4
1-4,6,8
```

When participants are omitted, workflows that require main and cue data search
for ids that have both `Part*Data.mat` and `Part*CueData.mat` available.
