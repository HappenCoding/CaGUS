# CaGUS Experimental Repository

This repository contains a lightweight, GitHub-friendly subset of the CaGUS experimental data and analysis outputs. The full local dataset is not included. Each experimental condition keeps at most one representative raw video, while detection reports and result tables are preserved when available.

## Structure

- `data/`: English experiment hierarchy with representative raw videos, `detection_report.md`, `detection_results`, `trajectory_nte_test.md`, and `trajectory_nte_test_results`.
- `scripts/`: differential detection, capability-boundary analysis, automatic boundary discovery, and geometry-conditioned transport projection scripts.
- `data/global_detection_summary.md`: repository-level detection summary generated from condition summaries.
- `repository_manifest.csv`: list of copied conditions and selected representative videos.

## Data Policy

Only one raw video is included per condition folder to keep the repository small. PDF files and unrelated standalone figures are intentionally excluded. Video files are tracked through Git LFS because at least one representative raw video is larger than 100 MB.

Generated on 2026-09-01 15:39:56.
