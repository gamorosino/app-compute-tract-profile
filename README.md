# Tract Profile Computation (app-compute-tract-profile)

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
| `smooth_density` | bool | `--smooth-density` (flag if true) |
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
