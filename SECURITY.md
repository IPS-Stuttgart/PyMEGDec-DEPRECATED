# Security policy

PyMEGDec is deprecated and kept only as a compatibility and reproducibility
archive for historical BUSH-MEG analyses. NeuRepTrace is the maintained successor
for reusable M/EEG decoding functionality.

## Supported versions

Only the archived `main` branch is considered for narrow compatibility and
reproducibility fixes. Feature work, new datasets, new decoders, and reusable
workflow improvements should be made in NeuRepTrace instead.

## Reporting a vulnerability

If a vulnerability affects reusable decoding, dataset loading, metrics,
probability-observation processing, or reporting code that now belongs in
NeuRepTrace, report it in NeuRepTrace.

Open a PyMEGDec issue only when the vulnerability is specific to this archive,
for example package metadata, historical compatibility wrappers, or a legacy
script that must remain usable for reproducibility. Avoid including private MEG
data or credentials in public reports.

## Expected response

Because this repository is an archive, fixes will normally be limited to:

- dependency or packaging metadata needed for safe historical installation;
- documentation updates that prevent unsafe use;
- compatibility changes that route affected functionality to NeuRepTrace;
- minimal changes that keep historical reproduction commands runnable.
