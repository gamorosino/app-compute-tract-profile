# Tract Profile Computation (app-compute-tract-profile)

<!-- brainlife-app:begin (generated from the App manifest; edit the manifest, not this block) -->

> **Computes along-tract profiles of diffusion metrics (FA, MD, RD, AD) for a tractogram with DIPY, using optional centroid, medoid or skeleton reference curves, and outputs tractmeasures CSVs and profile plots.**

Resamples each streamline to n_points nodes, orients them against a reference curve (QuickBundles/MBKM centroid or medoid, a tracklib density skeleton, or its nearest real streamline), samples the scalar maps along the nodes with optional distance weighting, and computes a dispersion measure. In the brainlife configuration it runs once per tensor metric (fa, md, ad, rd) and merges the results. Use it to compare microstructure along a bundle across subjects; the tractogram should contain a single bundle in the same space as the tensor maps.

| | |
|---|---|
| 📥 **Inputs** | `neuro/track/tck` · `neuro/tensor` · `neuro/anat/t1w` (optional) |
| 📤 **Outputs** | `generic/images` · `neuro/tractmeasures` |
| 🐳 **Container** | `gamorosino/tract_align:v1.0` |
| ⚙️ **Resources** | 1 CPUs · 1 h walltime |
| 🔗 **Run it** | [doi.org/10.25663/brainlife.app.911](https://doi.org/10.25663/brainlife.app.911) |

#### 📥 Inputs

| Input | Datatype | Required | Description |
|---|---|---|---|
| `tck` | `neuro/track/tck` | yes | Tractogram of a single white matter bundle (.tck). Must be in the same space as the tensor maps and the reference T1. |
| `tensor` | `neuro/tensor` | yes | Tensor (DTI) dataset; its FA, MD, AD and RD maps are profiled. |
| `reference` | `neuro/anat/t1w` | no | T1-weighted image used as spatial reference for loading .tck streamlines. |

#### 📤 Outputs

| Output | Datatype | Description |
|---|---|---|
| `figures` | `generic/images` | PNG plots of the tract profiles (and reference curve views). |
| `tractmeasures` | `neuro/tractmeasures` | Tractmeasures CSV with the profile of each metric at every node. |

<details>
<summary><b>⚙️ Parameters</b> (9)</summary>

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_points` | number | `100` | Number of nodes each streamline is resampled to along the tract. |
| `centroid_method` | enum | `skeleton` | Reference curve used to orient streamlines and weight nodes: none, QuickBundles centroid or medoid, tracklib density skeleton, or the real streamline nearest to the skeleton. Options: `none`, `centroid-qb`, `medoid-qb`, `skeleton`, `nn_skeleton`. |
| `weight_node_by_centroid` | boolean | `true` | If true, weights each node's contribution by its distance to the reference curve (--weight-node-by-centroid). |
| `dispersion_metric` | enum | `mad` | Dispersion measure computed across streamlines at each node: std, cv, var or mad. Options: `std`, `cv`, `var`, `mad`. |
| `color_map` | string | `jet` | Matplotlib colormap used in the profile plots. |
| `perc` | number | `0.95` | Percentile of the voxel density used to extract the skeleton backbone (skeleton methods). |
| `smooth_density` | boolean | `false` | If true, applies Gaussian smoothing to the voxel density map before skeleton extraction. |
| `length_thr` | number | `0.25` | Minimum length fraction for a backbone segment to be kept (skeleton methods). |
| `no_legend` | boolean | `true` | If true, hides the legend in the profile plots. |

</details>

📚 **References used by this App:**
- Hayashi S, et al. brainlife.io: a decentralized and open-source cloud platform to support neuroscience research. Nat Methods 21(5):809-813 (2024). [doi:10.1038/s41592-024-02237-2](https://doi.org/10.1038/s41592-024-02237-2)

<!-- brainlife-app:end -->

`app-compute-tract-profile` computes diffusion metric/scalar profiles along white matter tracts using `compute_tract_profile.py` (tract profiling with optional centroid/medoid/skeleton reference curves, weighting strategies, dispersion, and multiple plotting outputs).

Repository contents:
- `compute_tract_profile.py`: the Python CLI that does the computation and plotting.
- `main`: a bash wrapper that reads `config.json` and forwards options to `compute_tract_profile.py` inside the container `docker://gamorosino/tract_align:latest`.

---

## Author

**Gabriele Amorosino**  
Email: gabriele.amorosino@utexas.edu

---

## Usage

### Running on Brainlife.io
Use Brainlife UI / CLI as usual. (Note: Brainlife UI currently exposes only a subset of the options; advanced options can be set via `config.json` when running locally or if your platform supports custom config injection.)

### Running locally

#### Prerequisites
- Singularity
- `jq`

#### Steps
```bash
git clone https://github.com/gamorosino/app-compute-tract-profile.git
cd app-compute-tract-profile
chmod +x main
./main
```

By default, `main` reads `config.json` in the current directory. You can also set:
```bash
CONFIG=/path/to/config.json ./main
```

---

## Execution modes (important)

The wrapper supports **two modes**, depending on which keys you put in `config.json`:

### Mode A — Legacy “one run per metric” (backward compatible)
If you do **not** specify `scalar` or `scalars`, and you provide one or more of:
- `fa`, `md`, `ad`, `rd`

then `main` will run `compute_tract_profile.py` **once per metric**, writing to:
- `./output/fa`, `./output/md`, etc.

This preserves the app’s original behavior.

### Mode B — Single-run “full CLI pass-through”
If you provide either:
- `scalar` (single scalar map), **or**
- `scalars` (array of scalar maps; typically for multi-tract mode),

then `main` runs `compute_tract_profile.py` **once** and passes through all configured options.

---

## `config.json` keys and CLI mapping

All keys are **snake_case**. The wrapper translates them to the Python CLI flags as shown below.

### 1) Tractogram inputs

| config key | Type | CLI flag | Notes |
|---|---:|---|---|
| `tck` | string | (used to form `--tract`) | Optional if `trk` or `tracts` is used |
| `trk` | string | (used to form `--tract`) | Optional if `tck` or `tracts` is used |
| `tracts` | array[string] | `--tracts` | Multi-tract mode (Python uses this instead of `--tract`) |
| `tract_labels` | array[string] | `--tract-labels` | Must match number of `tracts` |

### 2) Scalar inputs

| config key | Type | CLI flag | Notes |
|---|---:|---|---|
| `scalar` | string | `--scalar` | Triggers single-run mode |
| `scalars` | array[string] | `--scalars` | Triggers single-run mode; must match `tracts` length |
| `fa` | string | legacy Mode A | Used only in legacy per-metric loop |
| `md` | string | legacy Mode A | Used only in legacy per-metric loop |
| `ad` | string | legacy Mode A | Used only in legacy per-metric loop |
| `rd` | string | legacy Mode A | Used only in legacy per-metric loop |

### 3) Reference images

| config key | Type | CLI flag | Notes |
|---|---:|---|---|
| `reference_img` | string | `--reference-img` | Preferred single reference image |
| `t1` | string | `--reference-img` | Backward compatible alias for reference image |
| `references` | array[string] | `--references` | Per-tract references; must match `tracts` length |

### 4) Core options

| config key | Type | CLI flag | Default in python |
|---|---:|---|---|
| `output` | string | `--output` | `tract_profile` (python default) |
| `n_points` | int | `--n_points` | `100` |
| `centroid_path` | string | `--centroid-path` | `null` |
| `centroid_method` | string | `--centroid-method` | `none` |

Valid `centroid_method` values:
- `none`
- `centroid-qb`
- `centroid-mbkm`
- `medoid-qb`
- `medoid-mbkm`
- `skeleton`
- `nn_skeleton`
- `all`

### 5) Skeleton/backbone (tracklib) options

| config key | Type | CLI flag |
|---|---:|---|
| `perc` | float | `--perc` |
| `smooth_density` | bool | `--smooth-density` (flag if false) |
| `length_thr` | float | `--length-thr` |

### 6) MBKM options

| config key | Type | CLI flag |
|---|---:|---|
| `n_clusters` | int | `--n-clusters` |
| `num_prototypes` | int | `--num-prototypes` |

### 7) Weighting options

| config key | Type | CLI flag |
|---|---:|---|
| `weights_std` | float | `--weights-std` |
| `no_weights` | bool | `--no-weights` (flag if true) |
| `weight_by_centroid` | bool | `--weight-by-centroid` (flag if true) |
| `weight_node_by_centroid` | bool | `--weight-node-by-centroid` (flag if true) |
| `no_weight_by_centroid_curve` | bool | `--no-weight-by-centroid-curve` (flag if true) |
| `average_centroid` | bool | `--average-centroid` (flag if true) |

### 8) Dispersion options

| config key | Type | CLI flag |
|---|---:|---|
| `dispersion_metric` | string | `--dispersion-metric` (`std`, `cv`, `var`, `mad`) |
| `weighted_dispersion` | bool | `--weighted-dispersion` (flag if true) |
| `no_dispersion` | bool | `--no-dispersion` (flag if true) |

### 9) Plot styling options

| config key | Type | CLI flag |
|---|---:|---|
| `colors` | string | `--colors` (comma-separated string like `"red,blue"`) |
| `color_map` | string | `--color-map` |
| `y_axis_range` | string | `--y-axis-range` (e.g. `"0,0.2"`) |
| `ylabel` | string | `--ylabel` |
| `line_width` | float | `--line-width` |
| `no_legend` | bool | `--no-legend` (flag if true) |

Figure sizes (strings `"W,H"`):
- `profile_figure_size` → `--profile-figure-size`
- `boxplot_figure_size` → `--boxplot-figure-size`
- `matrix_figure_size` → `--matrix-figure-size`
- `hist_figure_size` → `--hist-figure-size`
- `ref3d_figure_size` → `--ref3d-figure-size`

Fonts:
- `tract_profile_font_size` → `--tract-profile-font-size`
- `boxplot_font_size` → `--boxplot-font-size`

### 10) Boxplot summary options

| config key | Type | CLI flag |
|---|---:|---|
| `boxplot_summary` | bool | `--boxplot-summary` (flag if true) |
| `boxplot_stat` | string | `--boxplot-stat` (`profile` or `mean`) |
| `boxplot_title` | string | `--boxplot-title` |
| `boxplot_x_label` | string | `--boxplot-x-label` |
| `boxplot_width` | float | `--boxplot-width` |
| `boxplot_spacing` | float | `--boxplot-spacing` |
| `boxplot_err` | string | `--boxplot-err` (`none`, `std`, `sem`) |
| `boxplot_y_axis_range` | string | `--boxplot-y-axis-range` |

---

## Example configs

### Example 1 — Legacy FA/MD mode (per-metric loop)
```json
{
  "tck": "/path/to/tractogram.tck",
  "fa": "/path/to/fa.nii.gz",
  "md": "/path/to/md.nii.gz",
  "t1": "/path/to/t1w.nii.gz",
  "n_points": 100
}
```

### Example 2 — Single-run with one scalar + centroid method
```json
{
  "trk": "/path/to/bundle.trk",
  "scalar": "/path/to/FA.nii.gz",
  "reference_img": "/path/to/t1w.nii.gz",
  "centroid_method": "centroid-qb",
  "weight_by_centroid": true,
  "dispersion_metric": "std",
  "n_points": 100,
  "output": "./output/bundle_FA"
}
```

### Example 3 — Multi-tract mode (one scalar per tract)
```json
{
  "tracts": ["/path/to/CST_L.trk", "/path/to/CST_R.trk"],
  "scalars": ["/path/to/FA_L.nii.gz", "/path/to/FA_R.nii.gz"],
  "references": ["/path/to/ref_L.nii.gz", "/path/to/ref_R.nii.gz"],
  "tract_labels": ["CST_L", "CST_R"],
  "centroid_method": "all",
  "average_centroid": true,
  "boxplot_summary": true,
  "boxplot_stat": "profile",
  "output": "./output/group_FA",
  "n_points": 100
}
```

---

## Outputs

Outputs are written under `./output/` by default (and/or wherever `output` points). Depending on options, the Python script may generate:
- `.npz` profile files
- `.png` plots (profiles, multi-tract overlay, similarity matrix, histograms, 3D reference views)
- `.trk` reference streamline files
- `.csv` summary comparisons (multi-tract)

---

## Container

This app runs:
```bash
singularity exec -e \
  docker://gamorosino/tract_align:latest \
  micromamba run -n tract_align python compute_tract_profile.py [args]
```

---

## Citation

If you use this app in your research, please cite:
- **Brainlife.io**: Hayashi, S., et al. (2024). *Nature Methods, 21*(5), 809–813. DOI: 10.1038/s41592-024-02237-2
