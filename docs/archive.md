# Archive policy

PyMEGDec is a deprecated compatibility and reproducibility archive for historical BUSH-MEG / PyMEGDec analyses. Use [NeuRepTrace](https://github.com/IPS-Stuttgart/NeuRepTrace) for new dataset loading, decoding, evaluation, diagnostics, reporting, benchmarking, and probability-trace work.

## Maintained scope

Changes are appropriate here only when they support one of these archive-maintenance goals:

- keep historical PyMEGDec commands installable and runnable;
- preserve or document exact old output schemas used by existing papers, notebooks, or reports;
- fix security, packaging, CI, or dependency breakage that blocks reproduction;
- keep thin compatibility wrappers aligned with NeuRepTrace command names; or
- clarify how to migrate a legacy PyMEGDec workflow to NeuRepTrace.

Do not add new reusable scientific methods, new benchmarks, new generic diagnostics, new model-selection procedures, or new report generators to PyMEGDec. Implement those in NeuRepTrace and call them from this repository only through compatibility wrappers when old command names must remain available.

## Repository ownership boundary

| Belongs in NeuRepTrace | May remain in PyMEGDec |
| --- | --- |
| Generic feature-matrix decoding | Historical `Part*Data.mat` / `Part*CueData.mat` conventions |
| MNE and FieldTrip-style reusable loaders | BUSH-MEG-specific file-role metadata and output names |
| Classifier calibration and prediction-table diagnostics | Legacy command aliases and compatibility wrappers |
| Temporal generalization, onset/state inference, and probability-observation validation | Paper-specific alpha-band, CTF-geometry, reaction-time, and export scripts |
| Generic report aggregation and reusable plots | Notes needed to reproduce archived CSV artifacts |

## Final validation checklist

Before merging archive-maintenance changes, verify the following where applicable:

1. Fast tests still run without private MEG data:

   ```bash
   python -m unittest discover -v
   ```

2. NeuRepTrace migration specs still validate:

   ```bash
   neureptrace dataset validate dataset.yml
   ```

3. Compatibility shims still make the intended ownership boundary explicit. A PyMEGDec wrapper should either call a NeuRepTrace implementation or state why it remains legacy-only.

4. Any data-dependent reproduction command documents the required private input files and expected output table names.

5. New documentation points readers to NeuRepTrace for new work.

## Closing the archive

Once downstream scripts no longer require active PyMEGDec changes, use GitHub's repository archive setting. After archival, the repository should be read-only and the canonical development location should be NeuRepTrace.

Suggested final order:

1. Merge the final archive-closure PR.
2. Confirm CI and documentation builds pass.
3. Confirm the latest branch contains the deprecation notice, archive policy, migration instructions, and package-version metadata.
4. Archive the repository in GitHub settings.
