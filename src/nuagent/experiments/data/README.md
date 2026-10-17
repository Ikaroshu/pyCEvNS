# COHERENT CsI 2017 data

Source: [Zenodo record 1228631](https://zenodo.org/records/1228631), DOI
10.5281/zenodo.1228631; description [arXiv:1804.09459](https://arxiv.org/abs/1804.09459).
Converted on 2026-10-17 by `convert_coherent_csi.py`.

`coherent_csi_release/` retains the needed release files. A provenance comment
was prepended to each machine-readable file; original contents follow unchanged.
The release README explains the inputs and `coherent_parameters.yaml` specifies
the response and normalization. No download or YAML dependency is needed to run:

```sh
uv run python src/nuagent/experiments/data/convert_coherent_csi.py
```

`coherent_csi_2017.csv` contains 12 bins with edges 6,8,...,30 PE. Observed and
steady-state counts are raw beam-on coincidence and anti-coincidence counts,
respectively, summed over the **full released 0 <= time < 12 us** window (24
centers 0.25,...,11.75 us). Both windows have the same exposure and duration, so
the AC scale is 1. The observed total is 855 and the AC total is 703. Their
selection efficiency is already applied in the release. The AC normalization
prior is 1/sqrt(703); its per-bin statistical noise is treated as exact.

The prompt neutron distribution is counts per 1-PE bin per GWhr **before**
acceptance. The script applies the released efficiency at each 1-PE bin center,
multiplies by the released 7.47594 GWhr exposure, then sums into 2-PE bins. The
total is 5.419341500706415, with a 25% normalization prior. Full timing PDFs
integrate to one, so no timing rescaling is applied. The signal prior is 28%.

This timing choice differs from the publication's likelihood selection
(6 <= PE < 30, time < 6 us; arXiv:1708.01294, supplemental statistical analysis).
This prototype integrates timing out as specified; it does not reproduce that
two-dimensional likelihood. Signal folding uses the legacy continuous recoil
response and constant quenching, without photoelectron smearing. The 173 +/- 48
published signal expectation is a coarse normalization check, not an exact
reproduction target. The legacy README recoil window is a separate regression.
