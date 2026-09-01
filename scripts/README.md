# Scripts

## Detection and Capability Analysis

`run_mobicom_detection.py` runs the wall-video differential detection pipeline. By default it reads the repository `data` directory:

```bash
python scripts/run_mobicom_detection.py
```

`run_angle_distance_detection.py` produces the distance-angle heatmap workflow. `analyze_capability_boundary.py` aggregates existing condition summaries into a CSV and a factor-level capability table:

```bash
python scripts/analyze_capability_boundary.py
```

## Geometry-Conditioned Transport

`run_transport_boundary_discovery.py` implements Appendix B boundary discovery. It detects line candidates on the visible wall, constructs two-sided observation patches, and keeps boundaries with above-background temporal perturbation energy:

```bash
python scripts/run_transport_boundary_discovery.py data/distance/850/img_8941.mov
```

`run_geometry_projection.py` implements the analytical projection in Section 2.3 and Appendix A. It builds a first-order planar kernel, projects each wall-frame perturbation with the kernel adjoint, and saves a hidden-space activity response, peak trajectory, and PSR statistics:

```bash
python scripts/run_geometry_projection.py data/distance/850/img_8941.mov scripts/example_transport_geometry.json
```

The example geometry uses the measurement coordinates from the controlled setup. Adapt the wall and hidden-region bounds, grid sizes, and occluding segments to the experiment before interpreting the resulting trajectory. The projected response is a geometry-conditioned activity representation, not a reconstructed hidden image.

`transport_kernel.py` is the shared implementation of the discrete kernel, adjoint projection, multi-response fusion, and PSR calculation.
