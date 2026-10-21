# Corrected-gold angle and mesh controls

Native 3D FDTD study of one fixed five-length tetramer geometry with independent odd/even quadrant angles. This folder contains twelve completed native cases, full R/T/A spectra, two normalized electric-field planes at 23 wavelengths, native material fits, PML/grid audits and the analysis summary. Large native FSP files remain local and are not tracked.

The production package and its 2.2.0 wheel are unchanged. The metal model passes its material gates, but **the full metal-coated device has not passed the agreed mesh-convergence criteria**. The 8/4 nm and 4/2 nm local-metal comparisons fail; changing conformal variants or aligning the metal interfaces does not establish convergence. The direct-glass-contact control passes its 8/4 nm checks.

The mirror mapping `(theta_odd, theta_even) -> (-theta_even, -theta_odd)` preserves the measured scalar R/T/A for the tested normal, x-polarized illumination. Its maximum native R discrepancy is approximately 1.7e-8. The four angle endpoints show a significant depth change, but their quantitative values still require fine-grid confirmation.

Run a case on Windows with the installed Lumerical API and numpy/psutil:

```powershell
python scripts/study_rotation_jc220.py --config tmp/jc220_native/job.json --case au_both_8
```

The config must be a newly initialized Au100/JC metal-only job; the source job used here has the default seed and bounds. To initialize it elsewhere, use `python -m deta_generator2 init --job-dir jobs/rotation220 --preset au-metal --samples 12 --threads 8`, then pass that new config path. Completed matching cases are preserved.

Postprocessing additionally uses scipy and matplotlib:

```powershell
python scripts/analyze_rotation_jc220.py
```

The analysis includes the independent CV0 planar checks under `research/metal_loss_fit/film_benchmark_clear_pml/`. Its valley width is measured at half prominence, not equated to modal Q. Field means and percentiles use area weights and incident source intensity in CWnorm. Source/PML and numerical convergence checks are recorded separately.

The next single-sample reference is packaged in `artifacts/jc220_global4_metal2_reference.zip`: global maximum 4 nm, local-metal z maximum 2 nm, identical geometry, one sample, 32 threads. It has not been solved locally because of available memory. The full report and server instructions are in [Report](https://github.com/CypherZzzyer05/Report/blob/main/reports/2026-10-22_rotation_jc220/README.md).
