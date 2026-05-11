# Run Order and Artifact Guide

## Recommended order

1. Start with the 250,000-timestep notebook for a fast validation run.
2. Move to the 500,000 or 1,000,000-timestep notebook if the result needs a longer training budget.
3. Use the 3,000,000-timestep notebook only when sufficient runtime is available.

## Core run flow inside each notebook

1. Run environment setup and imports.
2. Run environment validation cells.
3. Train or load the Section 01 Hierarchical PPO checkpoint.
4. Run the comparison training section if fair model comparison is required.
5. Run evaluation and summary-table cells.
6. Run video export cells after checkpoints or `trained_models` are available.
7. Run the artifact manifest/bundle cell to collect `.zip`, `.pt`, `.csv`, plots, and videos.

## Presentation note

The comparison chart should be reported as the quantitative result. Gameplay videos are single-seed demonstrations and may score higher or lower than the mean comparison score.


## v9 note

Run the notebook from the top. The artifact tag is now available immediately after the import/path setup cell, so video export cells can create tagged filenames without waiting for the comparison section.


## v10 Video Export Flag Fix

This package defines `EXPORT_COMPARISON_3D_VIDEOS` near the start of each notebook and adds fallback definitions inside the video diagnostic cell. This prevents the video diagnostics from failing when the diagnostic cell is run before the comparison-video export cell.
