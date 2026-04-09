#!/usr/bin/env python3
"""
Compute tract profiles (e.g., FA) using DIPY with optional centroid/backbone strategies.

Supports QuickBundles (QB) and MBKM centroid/medoid extraction, Tracklib backbone,
and NN-skeleton (nearest real streamline to the backbone via nilab.nearest_neighbors).

Examples
--------
1) Simple weighted AFQ profile (Yeatman-style):
   python compute_tract_profile.py \
       --tract bundle.trk \
       --scalar FA.nii.gz \
       --output bundle_FA_profile

2) QuickBundles centroid to orient & as reference:
   python compute_tract_profile.py \
       --tract bundle.trk \
       --scalar FA.nii.gz \
       --centroid-method centroid-qb \
       --output bundle_FA_qb

3) MBKM medoid as reference:
   python compute_tract_profile.py \
       --tract bundle.trk \
       --scalar FA.nii.gz \
       --centroid-method medoid-mbkm \
       --output bundle_FA_mbkm

4) Full backbone (density-based skeleton from tracklib):
   python compute_tract_profile.py \
       --tract bundle.trk \
       --scalar FA.nii.gz \
       --centroid-method skeleton \
       --reference-img structural.nii.gz \
       --perc 0.75 --length-thr 0.9 \
       --output bundle_FA_skeleton
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

from dipy.io.image import load_nifti
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.tracking.streamline import set_number_of_points, Streamlines
from dipy.segment.clustering import QuickBundles
from dipy.segment.featurespeed import ResampleFeature
from dipy.segment.metricspeed import AveragePointwiseEuclideanMetric
import dipy.stats.analysis as dsa
from dipy.tracking.distances import bundles_distances_mam

from dipy.tracking.streamline import values_from_volume

from pathlib import Path

from itertools import combinations
from scipy.stats import pearsonr, spearmanr

from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import csv

def add_metric_to_tract_measure_map(
    tract_measure_map,
    subject_id,
    structure_id,
    metric_name,
    profile,
    dispersion,
    x_coords=None,
    y_coords=None,
    z_coords=None,
):
    for node in range(len(profile)):
        key = (structure_id, node + 1)

        if key not in tract_measure_map:
            tract_measure_map[key] = {
                "subjectID": subject_id,
                "structureID": structure_id,
                "nodeID": node + 1,
                "x_coords": x_coords[node] if x_coords is not None else "",
                "y_coords": y_coords[node] if y_coords is not None else "",
                "z_coords": z_coords[node] if z_coords is not None else "",
            }

        tract_measure_map[key][metric_name] = profile[node]
        tract_measure_map[key][f"{metric_name}_sd"] = dispersion[node] if dispersion is not None else ""

def nature_style_plot(
    ax,
    ymin=None,
    ymax=None,
    n_yticks=3,
    spine_width=2,
    tick_length=6,
    tick_width=2,
    fontsize=16,
    y_decimals=2,
    add_origin_padding=True, 
    pad_fraction=0.02
):
    """
    Apply Nature-style formatting to matplotlib axes.

    Includes:
    - left/bottom spines only
    - outward ticks
    - optional sparse y-ticks
    - trimmed spines for L-style axis ending
    """
    # Remove top/right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Style spines
    ax.spines["left"].set_linewidth(spine_width)
    ax.spines["bottom"].set_linewidth(spine_width)

    # Tick style
    ax.tick_params(
        axis="both",
        direction="out",
        length=tick_length,
        width=tick_width,
        labelsize=fontsize
    )

    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    # Y limits
    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)

    # Optional y-tick control
    if n_yticks is not None:
        ymin_current, ymax_current = ax.get_ylim()

        if n_yticks == 3:
            mid = (ymin_current + ymax_current) / 2
            ticks = [ymin_current, mid, ymax_current]
        elif n_yticks == 2:
            ticks = [ymin_current, ymax_current]
        else:
            ticks = np.linspace(ymin_current, ymax_current, n_yticks)

        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:.{y_decimals}f}" for t in ticks])

    # --- KEY: trim spines to tick range ---
    ymin_current, ymax_current = ax.get_ylim()
    ax.spines["left"].set_bounds(ymin_current, ymax_current)

    xticks = ax.get_xticks()
    if len(xticks) > 0:
        ax.spines["bottom"].set_bounds(xticks[0], xticks[-1])

    if add_origin_padding:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        xpad = pad_fraction * (xmax - xmin)
        ypad = pad_fraction * (ymax - ymin)

        ax.set_xlim(xmin - xpad, xmax)
        ax.set_ylim(ymin - ypad, ymax)

    return ax

def parse_figsize(figsize_str, default=(10, 6)):
    try:
        w, h = [float(v.strip()) for v in figsize_str.split(",")]
        return (w, h)
    except Exception:
        print(f"[WARN] Could not parse figure size '{figsize_str}', using default {default}")
        return default
def plot_profile_boxplot(profile_dict, labels, out_png,
                         ylabel="Scalar value",
                         title="Profile Summary Boxplot",
                         xlabel="Tract group",
                         colors=None,
                         no_legend=False,
                         ymin=None,
                         ymax=None,
                         fontsize=18,
                         boxplot_width=0.3,
                         boxplot_spacing=1.0,
                         figsize=(8, 6),
                         boxplot_err="none"):

    data = [np.asarray(profile_dict[label]).squeeze() for label in labels]

    fig, ax = plt.subplots(figsize=figsize)

    positions = 1 + np.arange(len(labels)) * boxplot_spacing

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=boxplot_width,
        patch_artist=True,
        showfliers=False
    )

    if colors is not None:
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    # --- ADD THIS BLOCK HERE ---
    if boxplot_err != "none":
        means = np.array([np.mean(d) for d in data])

        if boxplot_err == "std":
            errs = np.array([np.std(d, ddof=1) if len(d) > 1 else 0.0 for d in data])
        elif boxplot_err == "sem":
            errs = np.array([
                (np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
                for d in data
            ])

        ax.errorbar(
            positions,
            means,
            yerr=errs,
            fmt="o",
            color="black",
            markersize=4,
            capsize=4,
            elinewidth=1.5,
            linewidth=1.5,
            zorder=5
        )
    # --- END BLOCK ---

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=fontsize, rotation=20, ha="right")

    ax.set_xlim(
        positions[0] - boxplot_width * 0.9,
        positions[-1] + boxplot_width * 0.9
    )

    nature_style_plot(ax, ymin=ymin, ymax=ymax, fontsize=fontsize, n_yticks=3, y_decimals=2)

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[INFO] Saved boxplot summary → {out_png}")

def set_nature_style_yticks(ax, ymin=None, ymax=None):
    """
    Set elegant 3-tick y-axis style: [min, mid, max]
    """
    if ymin is None or ymax is None:
        ymin, ymax = ax.get_ylim()

    mid = (ymin + ymax) / 2
    ticks = [ymin, mid, ymax]

    ax.set_yticks(ticks)

def plot_colormap_profile(x, y, cmap_name="jet", lw=3):
    """
    Plot a line where color varies along the line according to y value.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=y.min(), vmax=y.max())

    lc = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidth=lw
    )

    lc.set_array(y[:-1])

    ax = plt.gca()
    ax.add_collection(lc)
    ax.autoscale()
    
    return lc

def plot_profile_similarity_matrix(all_profiles, labels, out_png, use_key="first", metric="pearson"):
    n = len(labels)
    mat = np.zeros((n, n), dtype=float)

    profile_list = []
    for _, profs in all_profiles.items():
        if use_key == "average" and "average" in profs:
            profile_list.append(np.asarray(profs["average"]).squeeze())
        else:
            key = list(profs.keys())[0]
            profile_list.append(np.asarray(profs[key]).squeeze())

    for i in range(n):
        for j in range(n):
            if metric == "pearson":
                mat[i, j] = pearsonr(profile_list[i], profile_list[j])[0]
            elif metric == "spearman":
                mat[i, j] = spearmanr(profile_list[i], profile_list[j])[0]
            else:
                raise ValueError(f"Unknown metric: {metric}")

    plt.figure(figsize=(8, 6))
    im = plt.imshow(mat, vmin=0.5, vmax=1, cmap="hot")
    plt.xticks(range(n), labels, rotation=45, ha="right", fontsize=12)
    plt.yticks(range(n), labels, fontsize=12)
    plt.colorbar(im, label=f"{metric} correlation")
    plt.title(f"Profile {metric.capitalize()} Similarity Matrix")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def compare_profiles(profile1, profile2):
    """Return a dictionary of similarity/difference metrics between two profiles."""
    p1 = np.asarray(profile1).squeeze()
    p2 = np.asarray(profile2).squeeze()

    if p1.shape != p2.shape:
        raise ValueError(f"Profile shapes differ: {p1.shape} vs {p2.shape}")

    pearson_r, pearson_p = pearsonr(p1, p2)
    spearman_rho, spearman_p = spearmanr(p1, p2)
    rmse = np.sqrt(np.mean((p1 - p2) ** 2))
    mae = np.mean(np.abs(p1 - p2))
    mean_diff = np.mean(p1 - p2)
    area_between = np.trapz(np.abs(p1 - p2))

    return {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
        "rmse": rmse,
        "mae": mae,
        "mean_diff": mean_diff,
        "area_between": area_between,
    }

def load_scalar_volume(scalar_path):
    """
    Load a scalar NIfTI image (e.g., FA, MD, etc.).
    Returns (volume, affine).
    """
    if not os.path.exists(scalar_path):
        raise FileNotFoundError(f"Scalar image not found: {scalar_path}")
    try:
        vol, vol_affine = load_nifti(scalar_path)
    except Exception:
        # fallback to nibabel if DIPY load_nifti fails
        img = nib.load(scalar_path)
        vol, vol_affine = np.asanyarray(img.dataobj), img.affine
    print(f"[INFO] Scalar volume loaded: {scalar_path}")
    return vol, vol_affine


def load_tractogram_with_space(tract_path, reference_img=None, n_points=None):
    """
    Load a tractogram (.trk or .tck) and return (StatefulTractogram, streamlines).
    Converts to RASMM space and optionally resamples to n_points.
    """
    ext = os.path.splitext(tract_path)[1].lower()

    # --- Handle reference logic properly ---
    if ext == ".trk":
        sft = load_tractogram(tract_path, "same", bbox_valid_check=False)
    else:
        if reference_img is None:
            raise ValueError(f"[ERROR] .tck files require --reference-img to be provided for {tract_path}")
        sft = load_tractogram(tract_path, reference_img, bbox_valid_check=False)

    sft.to_rasmm()
    streamlines = sft.streamlines

    if n_points is not None:
        streamlines = set_number_of_points(streamlines, n_points)

    return sft, streamlines

def compute_scalar_matrix(volume, streamlines, affine, n_points=100):
    """Return (n_streamlines, n_points) matrix of scalar values along each streamline."""
    streamlines_res = set_number_of_points(streamlines, n_points)
    values = values_from_volume(volume, streamlines_res, affine)
    return np.array(values)


def compute_dispersion_matrix(scalar_matrix, method='std', weights=None):
    """
    Compute node-wise dispersion (std, cv, var, or mad) across streamlines.

    If `weights` is provided, a weighted version of each metric is used.

    Parameters
    ----------
    scalar_matrix : array, shape (n_streamlines, n_points)
        Scalar values sampled along each streamline.
    method : str
        One of {'std', 'cv', 'var', 'mad'}.
    weights : array, optional, shape (n_streamlines, n_points)
        Node-wise weights for each streamline. If provided, dispersion is
        computed as weighted variance-like quantity.
    """
    if weights is None:
        if method == 'std':
            return np.std(scalar_matrix, axis=0)
        elif method == 'cv':
            mean = np.mean(scalar_matrix, axis=0)
            std = np.std(scalar_matrix, axis=0)
            return std / (mean + 1e-8)
        elif method == 'var':
            return np.var(scalar_matrix, axis=0)
        elif method == 'mad':
            med = np.median(scalar_matrix, axis=0)
            return np.median(np.abs(scalar_matrix - med), axis=0)
        else:
            raise ValueError(f"Unknown dispersion method: {method}")
    else:
        # Weighted dispersion
        w = np.asarray(weights)
        w /= np.sum(w, axis=0, keepdims=True) + 1e-12
        mean = np.sum(scalar_matrix * w, axis=0)
        var = np.sum(w * (scalar_matrix - mean)**2, axis=0)
        if method == 'std':
            return np.sqrt(var)
        elif method == 'var':
            return var
        elif method == 'cv':
            return np.sqrt(var) / (mean + 1e-8)
        elif method == 'mad':
            # Weighted median absolute deviation (approximate)
            med = np.sum(scalar_matrix * w, axis=0)
            mad = np.sum(np.abs(scalar_matrix - med) * w, axis=0)
            return mad
        else:
            raise ValueError(f"Unknown dispersion method: {method}")


# === PATHS / INTEGRATION ====================================================
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

TRACKLIB_PATH = os.path.abspath(
    os.path.join(SCRIPT_DIR, "libraries")
)
MBKM_PATH = os.path.abspath(
    os.path.join(SCRIPT_DIR, "../tractogram_alignment_repo/code/wm_registration")
)

for p in [TRACKLIB_PATH, MBKM_PATH]:
    if p not in sys.executable and p not in sys.path:
        sys.path.append(p)

# --- tracklib backbone ---
try:
    from tracklib import get_bundle_backbone
    TRACKLIB_AVAILABLE = True
    print(f"[INFO] tracklib successfully imported from: {TRACKLIB_PATH}")
except Exception as e:
    TRACKLIB_AVAILABLE = False
    print(f"[WARN] tracklib not found in {TRACKLIB_PATH}. Skeleton-based methods disabled.")
    print(f"[DEBUG] ImportError: {e}")

# --- MBKM clustering utilities ---
try:
    from dissimilarity_common import compute_dissimilarity
    from sklearn.cluster import MiniBatchKMeans
    MBKM_AVAILABLE = True
    print(f"[INFO] MBKM modules successfully imported from: {MBKM_PATH}")
except Exception as e:
    MBKM_AVAILABLE = False
    print(f"[WARN] MBKM modules not found in {MBKM_PATH}.")
    print(f"[DEBUG] ImportError: {e}")

# --- nilab nearest-neighbors for nn_skeleton ---
try:
    from nilab.nearest_neighbors import streamlines_neighbors
    NN_NEIGHBORS_AVAILABLE = True
    print("[INFO] nilab.nearest_neighbors imported successfully.")
except Exception as e:
    NN_NEIGHBORS_AVAILABLE = False
    print("[WARN] nilab.nearest_neighbors unavailable; nn_skeleton will fall back to Euclidean nearest.")
    print(f"[DEBUG] ImportError: {e}")


# === 3D VISUALIZATION UTILITIES ==============================================

COLORS_3D = {
    'centroid-qb': 'blue',
    'centroid-mbkm': 'purple',
    'medoid-qb': 'green',
    'medoid-mbkm': 'darkgreen',
    'skeleton': 'red',
    'nn_skeleton': 'orange',
    'average': 'black'
}

def plot_3d_refs(streamlines, refs_dict, out_prefix, max_display=300):
    """Generate 3D visualizations of streamlines and reference curves from multiple perspectives."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    subs = np.random.choice(len(streamlines), min(max_display, len(streamlines)), replace=False)
    views = [
        (90, -90, 'axial'),
        (0, 0, 'sagittal'),
        (0, 90, 'coronal'),
        (30, -60, 'oblique')
    ]
    for elev, azim, label in views:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for idx in subs:
            ax.plot(*streamlines[idx].T, color='gray', alpha=0.25)
        for name, ref in refs_dict.items():
            ax.plot(*ref.T, color=COLORS_3D.get(name, 'black'), linewidth=3, label=name)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"Reference curves ({label} view)")
        ax.legend(fontsize=20)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        plt.tight_layout()
        out_png = f"{out_prefix}_3D_{label}.png"
        plt.savefig(out_png, dpi=300)
        plt.close()
        print(f"[INFO] Saved 3D visualization ({label} view): {out_png}")


def plot_distance_histograms(streamlines, refs_dict, out_prefix, no_legend=False):
    """Plot histograms of mean streamline-to-reference distances."""
    n_streams = len(streamlines)
    distances = {}
    for name, ref in refs_dict.items():
        dists = []
        for sl in streamlines:
            dists.append(np.mean(np.linalg.norm(sl - ref, axis=1)))
        distances[name] = np.array(dists)
    plt.figure(figsize=(8, 5))
    for name, vals in distances.items():
        plt.hist(vals, bins=40, alpha=0.6, color=COLORS_3D.get(name, 'black'), label=name)
    plt.xlabel("Mean distance (mm)")
    plt.ylabel("Count")
    plt.title("Streamline-to-reference mean distances")
    if not no_legend: plt.legend(fontsize=20)
    plt.tight_layout()
    out_png = f"{out_prefix}_distance_histograms.png"
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[INFO] Saved distance histogram plot: {out_png}")


# === UTILITIES ==============================================================


def compute_nodewise_centroid_weights(streamlines, centroid, n_points=100):
    """
    Compute node-wise Mahalanobis weights relative to a fixed centroid curve.
    Each node's weights are normalized across streamlines.

    Parameters
    ----------
    streamlines : list of (N, 3) arrays
        Streamlines resampled to n_points.
    centroid : (n_points, 3) array
        Centroid curve used as reference.
    n_points : int
        Number of nodes to resample.
    Returns
    -------
    weights : (n_streamlines, n_points) array
    """
    from scipy.spatial.distance import mahalanobis

    streamlines = set_number_of_points(streamlines, n_points)
    centroid = set_number_of_points([centroid], n_points)[0]
    arr = np.asarray(streamlines)

    n_streams = len(arr)
    weights = np.zeros((n_streams, n_points))

    for n in range(n_points):
        node_coords = arr[:, n, :]
        mu = centroid[n]
        cov = np.cov(node_coords.T)
        # Regularize covariance if degenerate
        if np.linalg.det(cov) < 1e-10:
            cov += np.eye(3) * 1e-5
        inv_cov = np.linalg.inv(cov)
        for i in range(n_streams):
            d = mahalanobis(node_coords[i], mu, inv_cov)
            weights[i, n] = np.exp(-0.5 * (d ** 2))
        weights[:, n] /= np.sum(weights[:, n])
    return weights


def compute_centroid_distance_weights(streamlines, ref_curve, sigma=5.0):
    """
    Compute weights for each streamline based on its mean Euclidean distance
    to a reference curve (centroid/medoid/skeleton).

    Streamlines closer to the reference get higher weights, farther ones lower.
    """
    if ref_curve is None:
        raise ValueError("Reference curve required for --weight-by-centroid.")
    distances = []
    ref = np.asarray(ref_curve)
    for sl in streamlines:
        sl = np.asarray(sl)
        d = np.linalg.norm(sl - ref, axis=1).mean()
        distances.append(d)
    distances = np.array(distances)
    weights = np.exp(-0.5 * (distances / sigma) ** 2)
    weights /= np.sum(weights)
    return weights


def safe_load_streamlines(tract_path, reference="same", bbox_valid_check=False):
    """Version-robust loading for .trk/.tck using DIPY's load_tractogram."""
    ext = os.path.splitext(tract_path)[1].lower()
    if ext == ".trk":
        sft = load_tractogram(tract_path, reference, bbox_valid_check=bbox_valid_check)
    else:
        if reference == "same":
            raise ValueError("For non-.trk files, provide a NIfTI reference via --reference-img")
        ref_img = nib.load(reference)
        sft = load_tractogram(tract_path, ref_img, bbox_valid_check=bbox_valid_check)
    return sft


def load_reference_streamline(path_or_none, n_points=None):
    """Load a single-streamline .trk/.tck file; if multiple, use the first."""
    if path_or_none is None:
        return None
    sft = safe_load_streamlines(path_or_none, reference="same", bbox_valid_check=False)
    sl = sft.streamlines
    if len(sl) == 0:
        raise ValueError(f"No streamlines found in {path_or_none}")
    ref = np.asarray(sl[0])
    if n_points is not None:
        ref = np.asarray(set_number_of_points([ref], n_points)[0])
    return ref


def qb_centroid(streamlines, n_points=100):
    """QuickBundles centroid (AveragePointwiseEuclideanMetric)."""
    feature = ResampleFeature(nb_points=n_points)
    metric = AveragePointwiseEuclideanMetric(feature)
    qb = QuickBundles(threshold=np.inf, metric=metric)
    return np.asarray(qb.cluster(streamlines).centroids[0])


def medoid_by_centroid_distance(streamlines, centroid_curve):
    """Choose the real streamline with minimal L2 distance to centroid curve."""
    distances = [np.linalg.norm(np.asarray(sl) - centroid_curve) for sl in streamlines]
    return np.asarray(streamlines[int(np.argmin(distances))])


def compute_dissimilarity_matrix(streamlines, num_prototypes=40, verbose=False):
    """MBKM dissimilarity matrix via nilab/dissimilarity_common + MAM distance."""
    return compute_dissimilarity(
        streamlines,
        distance=bundles_distances_mam,
        num_prototypes=num_prototypes,
        prototype_policy='sff',
        verbose=verbose
    )


def mbkm_wrapper(tractogram, n_clusters=1, num_prototypes=40, verbose=False):
    """Run MBKM clustering and return (cluster_centers, medoid_indices)."""
    if not MBKM_AVAILABLE:
        raise ImportError("MBKM not available.")

    tractogram = np.array(tractogram)
    full_dissimilarity_matrix = compute_dissimilarity(
        tractogram,
        distance=bundles_distances_mam,
        num_prototypes=num_prototypes,
        prototype_policy='sff',
        verbose=verbose
    )
    streamlines_ids = np.arange(len(tractogram), dtype=np.int32)
    dissimilarity_matrix = full_dissimilarity_matrix[streamlines_ids]
    np.random.seed(42)
    mbkm = MiniBatchKMeans(
        init='random',
        n_clusters=n_clusters,
        batch_size=1000,
        n_init=10,
        max_no_improvement=5,
        verbose=0,
        random_state=42   
    )
    mbkm.fit(np.nan_to_num(dissimilarity_matrix))

    medoids_exhs = np.zeros(n_clusters, dtype=np.int32)
    for i, centroid in enumerate(mbkm.cluster_centers_):
        idx_i = np.where(mbkm.labels_ == i)[0]
        if idx_i.size == 0:
            idx_i = [0]
        tmp = full_dissimilarity_matrix[idx_i] - centroid
        medoids_exhs[i] = streamlines_ids[idx_i[(tmp * tmp).sum(1).argmin()]]

    return mbkm.cluster_centers_, medoids_exhs


def mbkm_centroid_medoid(streamlines, n_clusters=1, num_prototypes=40):
    """Return (centroid_curve, medoid_curve) from MBKM clustering."""
    centers, medoid_ids = mbkm_wrapper(streamlines, n_clusters=n_clusters, num_prototypes=num_prototypes)
    # Use the first cluster (default behavior)
    medoid_curve = np.asarray(streamlines[medoid_ids[0]])
    centroid_curve = np.mean(np.asarray(streamlines), axis=0)
    return centroid_curve, medoid_curve



def nearest_streamline_to_reference_nn(streamlines, reference_curve, n_points=100):
    """Find the real streamline nearest to `reference_curve` using nilab.nearest_neighbors.
    Handles all possible neighbor array shapes returned by streamlines_neighbors().
    """
    S = np.array(set_number_of_points(streamlines, n_points), dtype=object)
    R = np.array([set_number_of_points([reference_curve], n_points)[0]], dtype=object)
    dists, neigh = streamlines_neighbors(R, S, k=1)

    # Normalize neighbor indices
    try:
        idx = int(np.ravel(neigh)[0])
    except Exception:
        idx = int(neigh)
    return np.asarray(S[idx])



def orient_streamlines_consistent(streamlines, reference=None, n_pts=100):
    """
    Orient streamlines consistently:
      - If `reference` provided: orient each streamline to minimize L2 distance to it.
      - Fallback (when no reference): orient by bundle bounding-box origin (min corner).
    Also orients the reference curve to the same canonical direction if provided.
    Returns (streamlines_oriented, reference_oriented_or_None)
    """
    s = set_number_of_points(streamlines, n_pts)
    coords = np.vstack(s) if len(s) else np.zeros((1, 3))
    bbox_origin = coords.min(axis=0)

    ref = None
    if reference is not None:
        ref = np.asarray(set_number_of_points([reference], n_pts)[0])
        # orient ref to bbox-origin as well for consistency
        if np.linalg.norm(ref[-1] - bbox_origin) < np.linalg.norm(ref[0] - bbox_origin):
            ref = ref[::-1]

    out_list = []
    for sl in s:
        a = np.asarray(sl)
        flip_bbox = np.linalg.norm(a[-1] - bbox_origin) < np.linalg.norm(a[0] - bbox_origin)
        flip_ref = False
        if ref is not None:
            d_orig = np.linalg.norm(a - ref)
            d_flip = np.linalg.norm(a[::-1] - ref)
            flip_ref = d_flip < d_orig
        flip = flip_ref or (ref is None and flip_bbox)
        if flip:
            a = a[::-1]
        out_list.append(a)

    return Streamlines(out_list), ref


def compute_weights(streamlines, n_points=100, std=None):
    """Version-robust Yeatman-style Gaussian weights (DIPY API changed around 1.8)."""
    try:
        # Old DIPY (<1.8) expects std kwarg:
        return dsa.gaussian_weights(streamlines, n_points=n_points, std=std)
    except TypeError:
        # Newer DIPY (>=1.8) removed std kwarg:
        return dsa.gaussian_weights(streamlines, n_points=n_points)


def plot_profile(profile, out_png, ylabel="Scalar value", title="Tract Profile"):
    x = np.arange(profile.shape[-1])
    plt.figure(figsize=(8, 5))
    plt.plot(x, profile)
    plt.xlabel("Node")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def compute_backbone_with_tracklib(track_filename, output_file, reference_img,
                                   perc=0.75, smooth_density=False, length_thr=0.9, n_points=100):
    """Thin wrapper around tracklib.get_bundle_backbone with safety and shape."""
    if not TRACKLIB_AVAILABLE:
        raise ImportError("tracklib is not available — cannot compute skeleton.")
    print("[INFO] Computing backbone with tracklib.get_bundle_backbone ...")
    backbone = get_bundle_backbone(
        track_filename,
        output_file,
        reference_img,
        N_points=n_points,
        perc=perc,
        smooth_density=smooth_density,
        length_thr=length_thr
    )
    return np.asarray(backbone).squeeze()


# === MAIN ===================================================================
def main():
    ap = argparse.ArgumentParser(description="Compute AFQ-style tract profiles with centroid/backbone options.")
    ap.add_argument(
        "--tract",
        required=False,
        help="Single input tractogram (.trk/.tck). Optional if --tracts is provided."
    )

    ap.add_argument(
    "--scalar",
    required=False,
    default=None,
    help="Scalar map NIfTI (e.g., FA.nii.gz). Optional if --scalars is provided.")
    ap.add_argument(
    "--scalars",
    nargs="+",
    help="Optional list of scalar maps, one per tract. Overrides --scalar in multi-tract mode.")
    ap.add_argument("--output", default="tract_profile", help="Output prefix")
    ap.add_argument("--n_points", type=int, default=100, help="Nodes along tract")
    ap.add_argument("--centroid-path", default=None,
                    help="Path to a .trk/.tck containing a single reference streamline (overrides centroid-method).")
    ap.add_argument("--reference-img", default="same",
                    help="Reference NIfTI for non-.trk tractograms and backbone extraction.")
    ap.add_argument(
        "--references",
        nargs="+",
        help="Optional list of reference images, one per tract. Overrides --reference-img in multi-tract mode."
    )   
    ap.add_argument("--perc", type=float, default=0.25, help="Percentile for backbone extraction (default=0.25).")
    ap.add_argument("--smooth-density", action="store_true",
                    help="Enable Gaussian smoothing of voxel density map (default: OFF).")
    ap.add_argument("--length-thr", type=float, default=0.25,
                    help="Min length fraction for backbone retainment (default=0.25).")
    ap.add_argument("--n-clusters", type=int, default=1,
                    help="MBKM: number of clusters (default 1).")
    ap.add_argument("--num-prototypes", type=int, default=40,
                    help="MBKM: number of prototypes for dissimilarity computation (default 40).")
    ap.add_argument("--weights-std", type=float, default=None,
                    help="Yeatman Gaussian weighting std; leave None for DIPY default.")
    ap.add_argument("--no-weights", action="store_true",
                    help="Disable Gaussian weights (unweighted mean).")
    ap.add_argument("--weight-by-centroid", action="store_true",
                help="Weight streamlines by distance to centroid/skeleton instead of Gaussian node weights.")
    ap.add_argument("--centroid-method",
                choices=["none", "centroid-qb", "centroid-mbkm", "medoid-qb", "medoid-mbkm",
                         "skeleton", "nn_skeleton", "all"],
                default="none", help="How to derive a reference curve. Use 'all' to compute all methods.")
    ap.add_argument("--average-centroid", action="store_true",
                help="If used with --centroid-method all, computes and overlays the average centroid profile.")
    ap.add_argument("--no-weight-by-centroid-curve", action="store_true",
        help="If used with --weight-by-centroid or --no-weights, also compute and plot a "
            "traditional Yeatman-weighted tract profile in gray for visual comparison.")
    ap.add_argument("--dispersion-metric", choices=["std", "cv", "var", "mad"], default="std",
                help="Metric for dispersion along the tract.")
    ap.add_argument(
        "--weighted-dispersion", action="store_true",
        help="If set, compute dispersion using streamline weights "
            "(reflects uncertainty in the weighted mean rather than anatomical variability)."
    )
    ap.add_argument(
        "--no-dispersion", action="store_true",
        help="Disable computation and visualization of dispersion bands."
    )
    ap.add_argument(
        "--weight-node-by-centroid", action="store_true",
        help="Use node-wise Mahalanobis weighting around the selected centroid curve "
            "(AFQ-like, but referenced to the centroid rather than the bundle mean)."
    )
    ap.add_argument(
        "--tracts", nargs="+",
        help="List of multiple input bundles (.trk/.tck). Overrides --tract if provided."
    )
    ap.add_argument(
        "--colors", default=None,
        help="Comma-separated list of colors to use for plotting multiple tracts "
            "(e.g., 'red,blue,green'). Will cycle if fewer than bundles."
    )
    ap.add_argument(
        "--color-map",
        default=None,
        help="Colormap for single-tract profile lines (e.g., jet, viridis). "
            "If set, the line will be colored by scalar values along the tract."
    )
    ap.add_argument(
        "--tract-labels",
        nargs="+",
        help="Optional list of labels to use in the legend for each tract (must match number of tracts).",
    )
    ap.add_argument(
        "--y-axis-range",
        type=str,
        default=None,
        help="Optional y-axis limits as 'min,max' (e.g., 0,0.2)."
    )

    ap.add_argument(
        "--no-legend",
        action="store_true",
        help="Disable legend in plots."
    )
    ap.add_argument("--ylabel", default="Scalar value", help="Y label for plot.")

    ap.add_argument(
        "--line-width",
        type=float,
        default=3,
        help="Line width for tract profile plots."
    )
    ap.add_argument(
        "--boxplot-summary",
        action="store_true",
        help="Generate a boxplot summarizing tract profile values for each tract."
    )

    ap.add_argument(
        "--boxplot-stat",
        choices=["profile", "mean"],
        default="profile",
        help="How to summarize for the boxplot: "
            "'profile' uses all node values of each tract profile, "
            "'mean' uses the tract mean value as a single point."
    )

    ap.add_argument(
        "--boxplot-title",
        default="Tract Profile Summary",
        help="Title for the boxplot."
    )

    ap.add_argument(
        "--boxplot-x-label",
        default="Tract group",
        help="X-axis label for the boxplot."
    )
    ap.add_argument(
        "--boxplot-width",
        type=float,
        default=0.5,
        help="Width of each box in the summary boxplot."
    )

    ap.add_argument(
        "--boxplot-spacing",
        type=float,
        default=1.0,
        help="Spacing multiplier between adjacent boxes in the summary boxplot."
    )

    ap.add_argument("--profile-figure-size", type=str, default="10,6",
                    help="Figure size for tract profile plots as width,height.")
    ap.add_argument("--boxplot-figure-size", type=str, default="8,6",
                    help="Figure size for boxplots as width,height.")
    ap.add_argument("--matrix-figure-size", type=str, default="8,6",
                    help="Figure size for similarity matrix as width,height.")
    ap.add_argument("--hist-figure-size", type=str, default="8,5",
                    help="Figure size for histogram plots as width,height.")
    ap.add_argument("--ref3d-figure-size", type=str, default="10,8",
                    help="Figure size for 3D reference plots as width,height.")

    ap.add_argument(
        "--boxplot-err",
        choices=["none", "std", "sem"],
        default="none",
        help="Overlay mean ± error on the summary boxplot: standard deviation (std) or standard error of the mean (sem)."
    )

    ap.add_argument(
        "--tract-profile-font-size",
        type=float,
        default=24,
        help="Font size for tract profile plots."
    )
    ap.add_argument(
        "--boxplot-font-size",
        type=float,
        default=24,
        help="Font size for tract profile plots."
    )

    ap.add_argument(
        "--boxplot-y-axis-range",
        type=str,
        default=None,
        help="Optional y-axis limits as 'min,max' (e.g., 0,0.2)."
    )
    ap.add_argument("--subject-id", default="", help="Subject ID for tractmeasures.csv")
    args = ap.parse_args()


    base_dir = os.path.dirname(args.output)
    if base_dir == "":
        base_dir = "."

    figures_dir = os.path.join(base_dir, "figures")
    tractmeasures_dir = os.path.join(base_dir, "tractmeasures")
    work_dir = os.path.join(base_dir, "work")

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tractmeasures_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    output_stem = os.path.basename(args.output)
    work_prefix = os.path.join(work_dir, output_stem)
    fig_prefix = os.path.join(figures_dir, output_stem)
    tractmeasures_csv = os.path.join(tractmeasures_dir, "tractmeasures.csv")

    # Parse y-axis range if provided
    ymin, ymax = None, None
    if args.y_axis_range is not None:
        try:
            parts = [float(v.strip()) for v in args.y_axis_range.split(",")]
            if len(parts) == 2:
                ymin, ymax = parts
        except:
            print("[WARN] Could not parse --y-axis-range. Expected format: min,max")

    if args.boxplot_y_axis_range is not None:
        try:
            parts = [float(v.strip()) for v in args.boxplot_y_axis_range.split(",")]
            if len(parts) == 2:
                boxplot_ymin, boxplot_ymax = parts
        except:
            print("[WARN] Could not parse --boxplot-y-axis-range. Expected format: min,max")
    else:
        boxplot_ymin, boxplot_ymax =  ymin, ymax

    if args.tract is None and args.tracts is None:
        ap.error("You must provide either --tract or --tracts.")

    if args.scalar is None and args.scalars is None:
        ap.error("You must provide either --scalar or --scalars.")

    method_colors = {
        "centroid-qb": "tab:blue",
        "centroid-mbkm": "tab:orange",
        "medoid-qb": "tab:green",
        "medoid-mbkm": "tab:red",
        "skeleton": "tab:purple",
        "nn_skeleton": "tab:brown"
    }

    if args.tracts is not None:
        n_tracts = len(args.tracts)

        if args.scalars is not None and len(args.scalars) != n_tracts:
            ap.error(f"--scalars must have the same number of entries as --tracts ({n_tracts}).")

        if args.references is not None and len(args.references) != n_tracts:
            ap.error(f"--references must have the same number of entries as --tracts ({n_tracts}).")

    # --- Handle multiple tracts ---
    if args.tracts is not None:
        tract_paths = args.tracts
    else:
        tract_paths = [args.tract]

    # Per-tract scalar paths
    if args.scalars is not None:
        scalar_paths = args.scalars
    else:
        scalar_paths = [args.scalar] * len(tract_paths)

    # Per-tract reference images
    if args.references is not None:
        reference_paths = args.references
    else:
        reference_paths = [args.reference_img] * len(tract_paths)

    if args.colors is not None:
        color_list = [c.strip() for c in args.colors.split(",") if c.strip()]
    else:
        color_list = None


    profile_figsize = parse_figsize(args.profile_figure_size, default=(10, 6))
    boxplot_figsize = parse_figsize(args.boxplot_figure_size, default=(8, 6))
    matrix_figsize = parse_figsize(args.matrix_figure_size, default=(8, 6))
    hist_figsize = parse_figsize(args.hist_figure_size, default=(8, 5))
    ref3d_figsize = parse_figsize(args.ref3d_figure_size, default=(10, 8))
    # ============================================================
    # === MULTI-TRACT / MULTI-METHOD PROCESSING SECTION ===========
    # ============================================================

    all_profiles = {}
    all_ref_curves = {}
    tract_measure_map = {}
    # Determine tract list
    tract_paths = [args.tract] if args.tract else args.tracts
    tract_colors = []
    if args.colors:
        tract_colors = args.colors.split(",")
    else:
        # auto-generate distinct colors
        cmap = plt.get_cmap("tab10")
        tract_colors = [cmap(i % 10) for i in range(len(tract_paths))]



    # === Classic Yeatman fallback mode ===
    if args.centroid_method == "none":
        print("[INFO] Centroid method set to 'none' → using classic Yeatman AFQ-style profiles.")

        yeatman_profiles = {}
        colors = args.colors.split(",") if args.colors else None
        for i, tract_path in enumerate(tract_paths):
            scalar_path = scalar_paths[i]
            reference_path = reference_paths[i]

            print(f"[INFO] Processing (Yeatman) tract {i+1}/{len(tract_paths)}: {tract_path}")

            vol, vol_affine = load_scalar_volume(scalar_path)
            sft, streamlines = load_tractogram_with_space(tract_path, reference_path, n_points=args.n_points)

            # --- Compute classic Yeatman-style weights and profile ---
            streamlines_oriented = set_number_of_points(streamlines, args.n_points)
            weights = compute_weights(streamlines_oriented, n_points=args.n_points, std=args.weights_std)
            profile = dsa.afq_profile(vol, streamlines_oriented, vol_affine, weights=weights)
            yeatman_profiles[tract_path] = profile

            # --- Compute dispersion (optional) ---
            dispersion = None
            if not args.no_dispersion:
                print(f"[INFO] Computing dispersion for {tract_path}...")
                scalar_matrix = compute_scalar_matrix(vol, streamlines_oriented, vol_affine, n_points=args.n_points)
                dispersion = compute_dispersion_matrix(scalar_matrix, method=args.dispersion_metric)
                np.savez(f"{args.output}_{Path(tract_path).stem}_Yeatman_dispersion.npz", dispersion=dispersion)
            else:
                print(f"[INFO] Dispersion disabled (--no-dispersion).")
            
            metric_name = Path(scalar_path).stem.split(".")[0]

            add_metric_to_tract_measure_map(
                tract_measure_map=tract_measure_map,
                subject_id=args.subject_id,
                structure_id=Path(tract_path).stem,
                metric_name=metric_name,
                profile=profile,
                dispersion=dispersion,
                x_coords=None,
                y_coords=None,
                z_coords=None,
            )
                        
            # --- Save profile ---
            np.savez(f"{args.output}_{Path(tract_path).stem}_Yeatman.npz", profile=profile)
            print(f"[INFO] Saved Yeatman profile → {args.output}_{Path(tract_path).stem}_Yeatman.npz")

        # --- Plot all Yeatman profiles together ---
        plt.figure(figsize=(9, 5))
        for i, (tract, prof) in enumerate(yeatman_profiles.items()):
            color = colors[i % len(colors)] if colors else None
            label = Path(tract).stem
            x = np.arange(prof.shape[-1])
            plt.plot(x, prof, lw=args.line_width, label=label, color=color)

            disp_path = f"{args.output}_{Path(tract).stem}_Yeatman_dispersion.npz"
            if not args.no_dispersion and os.path.exists(disp_path):
                disp = np.load(disp_path)["dispersion"]
                plt.fill_between(x, prof - disp, prof + disp, color=color, alpha=0.2)

        plt.xlabel("Location")
        plt.ylabel(args.ylabel)
        if ymin is not None and ymax is not None:
            plt.ylim(ymin, ymax)
    
        plt.title("Classic Yeatman AFQ-style Tract Profiles")
        if not args.no_legend: plt.legend(fontsize=20)
        plt.tight_layout()
        plt.savefig(f"{fig_prefix}_Yeatman_profiles.png", dpi=300)
        plt.close()
        print(f"[INFO] Saved Yeatman overlay plot → {args.output}_Yeatman_profiles.png")
        
        base_dir = os.path.dirname(args.output)
        if base_dir == "":
          base_dir = "."
        
        tract_profiles_dir = os.path.join(base_dir, "tract_profiles")
        os.makedirs(tract_profiles_dir, exist_ok=True)
        
        csv_path = os.path.join(tractmeasures_dir, f"{metric_name}_tractmeasures.csv")

        rows = list(tract_measure_map.values())
        rows.sort(key=lambda r: (r["structureID"], r["nodeID"]))

        metric_names = sorted({
            k[:-3] if k.endswith("_sd") else k
            for row in rows
            for k in row.keys()
            if k not in {"subjectID","structureID","nodeID","x_coords","y_coords","z_coords"}
            and not k.endswith("_coords")
        })

        fieldnames = ["subjectID","structureID","nodeID"]

        for metric in metric_names:
            fieldnames.extend([metric,f"{metric}_sd"])

        fieldnames.extend(["x_coords","y_coords","z_coords"])

        with open(csv_path,"w",newline="") as f:
            writer = csv.DictWriter(f,fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"[INFO] Saved tract measures → {csv_path}")
        # skip centroid-related processing
        return


    # ------------------------------------------------------------
    for t_idx, tract_path in enumerate(tract_paths):
        scalar_path = scalar_paths[t_idx]
        reference_path = reference_paths[t_idx]
        tract_name = os.path.splitext(os.path.basename(tract_path))[0]
        color_main = tract_colors[t_idx]
        output_prefix = f"{work_prefix}_{tract_name}"
        print(f"\n[INFO] Processing tract: {tract_name}")

        # Load data
        print(f"[INFO] Loading tract: {tract_path}")
        sft, streamlines = load_tractogram_with_space(tract_path, reference_path, n_points=args.n_points)
        vol, vol_affine = load_scalar_volume(scalar_path)

        print(f"[INFO] Loaded {len(streamlines)} streamlines")


        # Determine which methods to run
        if args.centroid_method == "all":
            methods = ["centroid-qb", "centroid-mbkm", "medoid-qb",
                    "medoid-mbkm", "skeleton", "nn_skeleton"]
        else:
            methods = [args.centroid_method]

        profiles = {}
        ref_curves = {}

        # Handle optional custom labels for tracts
        if args.tract_labels:
            if len(args.tract_labels) != len(args.tracts):
                print(f"[WARN] Number of --tract-labels ({len(args.tract_labels)}) does not match number of --tracts ({len(args.tracts)}). Ignoring labels.")
                tract_labels = [Path(t).stem for t in args.tracts]
            else:
                tract_labels = args.tract_labels
        else:
            if args.tracts is not None:
                # Default: use tract filenames (without extension)
                tract_labels = [Path(t).stem for t in args.tracts]
            else:
                tract_labels = [Path(args.tract).stem]

        # --------------------------------------------------------
        for m in methods:
            print(f"[INFO] Deriving reference with method: {m}")
            streamlines_base = Streamlines(streamlines)  # fresh copy
            ref_curve = None

            # --- Compute reference curve ---
            if m == "centroid-qb":
                ref_curve = qb_centroid(streamlines_base, n_points=args.n_points)
            elif m == "centroid-mbkm":
                ref_curve, _ = mbkm_centroid_medoid(streamlines_base,
                                                    n_clusters=args.n_clusters,
                                                    num_prototypes=args.num_prototypes)
            elif m == "medoid-qb":
                c = qb_centroid(streamlines_base, n_points=args.n_points)
                ref_curve = medoid_by_centroid_distance(streamlines, c)
            elif m == "medoid-mbkm":
                _, ref_curve = mbkm_centroid_medoid(streamlines_base,
                                                    n_clusters=args.n_clusters,
                                                    num_prototypes=args.num_prototypes)
            elif m == "skeleton":
                ref_curve = compute_backbone_with_tracklib(
                    tract_path, f"{output_prefix}_{m}.trk", reference_path,
                    perc=args.perc, smooth_density=args.smooth_density,
                    length_thr=args.length_thr, n_points=args.n_points)
            elif m == "nn_skeleton":
                skeleton = compute_backbone_with_tracklib(
                    tract_path, f"{output_prefix}_{m}.trk", reference_path,
                    perc=args.perc, smooth_density=args.smooth_density,
                    length_thr=args.length_thr, n_points=args.n_points)
                if NN_NEIGHBORS_AVAILABLE:
                    ref_curve = nearest_streamline_to_reference_nn(streamlines_base,
                                                                skeleton,
                                                                n_points=args.n_points)
                else:
                    ref_curve = medoid_by_centroid_distance(streamlines_base, skeleton)

            # --- Orient streamlines ---
            streamlines_oriented, ref_curve_oriented = orient_streamlines_consistent(
                streamlines_base, reference=ref_curve, n_pts=args.n_points)
            if ref_curve_oriented is not None:
                ref_curve = ref_curve_oriented

            # --- Weight selection logic ---
            if args.no_weights:
                weights = None
            elif args.weight_node_by_centroid:
                print("[INFO] Using node-wise centroid-based Mahalanobis weighting (AFQ-like).")
                weights = compute_nodewise_centroid_weights(streamlines_oriented, ref_curve, n_points=args.n_points)
            elif args.weight_by_centroid:
                print("[INFO] Using global centroid-based streamline weighting.")
                weights = compute_centroid_distance_weights(streamlines_oriented, ref_curve)
            else:
                print("[INFO] Using traditional AFQ Yeatman-style weighting.")
                weights = compute_weights(streamlines_oriented, n_points=args.n_points, std=args.weights_std)

            # --- Compute scalar profile ---
            profile = dsa.afq_profile(vol, streamlines_oriented, vol_affine, weights=weights)
            profiles[m] = profile
            ref_curves[m] = ref_curve

            # --- Dispersion computation ---
            scalar_matrix = compute_scalar_matrix(vol, streamlines_oriented, vol_affine, n_points=args.n_points)
            dispersion = None
            if not args.no_dispersion:
                if args.weighted_dispersion and weights is not None:
                    print(f"[INFO] Computing weighted dispersion ({args.dispersion_metric})...")
                    dispersion = compute_dispersion_matrix(scalar_matrix,
                                                        method=args.dispersion_metric,
                                                        weights=weights)
                else:
                    dispersion = compute_dispersion_matrix(scalar_matrix,
                                                        method=args.dispersion_metric)
            else:
                print("[INFO] Dispersion computation disabled (--no-dispersion).")
            
            metric_name = Path(scalar_path).name.replace(".nii.gz","").replace(".nii","")
            add_metric_to_tract_measure_map(
              tract_measure_map=tract_measure_map,
              subject_id=args.subject_id,
              structure_id=tract_name,
              metric_name=metric_name,
              profile=profile,
              dispersion=dispersion,
              x_coords=ref_curve[:, 0] if ref_curve is not None else None,
              y_coords=ref_curve[:, 1] if ref_curve is not None else None,
              z_coords=ref_curve[:, 2] if ref_curve is not None else None,    )
            
            np.savez(f"{output_prefix}_{m}_dispersion.npz", dispersion=dispersion)
            profiles[f"{m}_dispersion"] = dispersion

            # --- Optional gray Yeatman-style line ---
            if args.no_weight_by_centroid_curve and (args.weight_by_centroid or args.no_weights or args.weight_node_by_centroid):
                gray_weights = compute_weights(streamlines_oriented, n_points=args.n_points, std=args.weights_std)
                gray_profile = dsa.afq_profile(vol, streamlines_oriented, vol_affine, weights=gray_weights)
                profiles["gray_reference"] = gray_profile

            # --- Save reference streamline ---
            ref_sft = StatefulTractogram([ref_curve], sft, Space.RASMM)
            save_tractogram(ref_sft, f"{output_prefix}_{m}_reference.trk", bbox_valid_check=False)
            np.savez(f"{output_prefix}_{m}.npz", profile=profile)

            print(f"[INFO] Saved → {output_prefix}_{m}.npz / _reference.trk")

        # --------------------------------------------------------
        # === Plot all profiles for this tract ===
        plt.figure(figsize=profile_figsize)
        ax = plt.gca()

        for m, prof in profiles.items():
            if m.endswith("_dispersion"):
                continue
            x = np.arange(prof.shape[-1])
            if m == "gray_reference":
                plt.plot(x, prof, color="gray", lw=1.5, linestyle="--", label="Yeatman-style")
                continue
            color = method_colors.get(m, color_main)
            use_colormap_line = (args.color_map is not None and len(methods) == 1)
            if use_colormap_line:
                lc = plot_colormap_profile(x, prof, cmap_name=args.color_map, lw=args.line_width)
                plt.plot([], [], color=color, lw=2, label=f"{tract_name}-{m}")
            else:
                plt.plot(x, prof, lw=args.line_width, label=f"{tract_name}-{m}", color=color)
             # Dispersion shading
            if not args.no_dispersion:
                disp_path = f"{output_prefix}_{m}_dispersion.npz"
                if os.path.exists(disp_path):
                    disp = np.load(disp_path)["dispersion"]
                    plt.fill_between(x, prof - disp, prof + disp, color=color, alpha=0.2)
        fontsize=args.tract_profile_font_size
        plt.xlabel("Position along tract (%)", fontsize=fontsize)
        plt.ylabel(args.ylabel, fontsize=fontsize)

        if ymin is not None and ymax is not None:
            plt.ylim(ymin, ymax)

        # explicit x-axis control
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])
        ax.set_xticklabels(["0", "50", "100"])
        nature_style_plot(ax, ymin, ymax, fontsize=fontsize,
                      add_origin_padding=True, 
                      pad_fraction=0.02)
        
        #set_nature_style_yticks(ax, ymin, ymax)
        #ax.spines["left"].set_linewidth(1.5)
        #ax.spines["bottom"].set_linewidth(1.5)
        #ax.tick_params(width=1.5, length=6)
        #plt.xticks(fontsize=20)
        #plt.yticks(fontsize=20)
        #plt.title(f"Tract Profiles (All Methods) — {tract_name}")
        if not args.no_legend: plt.legend(fontsize=fontsize)
        # --- Average centroid per tract ---
        if args.average_centroid:
            profile_keys = [k for k in profiles if not k.endswith("_dispersion") and k != "gray_reference"]
            mean_profile = np.mean(np.stack([profiles[k] for k in profile_keys]), axis=0)
            mean_ref = np.mean(np.stack(list(ref_curves.values())), axis=0)
            ref_sft = StatefulTractogram([mean_ref], sft, Space.RASMM)
            save_tractogram(ref_sft, f"{output_prefix}_average_reference.trk", bbox_valid_check=False)
            plt.plot(np.arange(mean_profile.shape[-1]), mean_profile, color="black", lw=2.5, label="average centroid")
            np.savez(f"{output_prefix}_average_centroid.npz", profile=mean_profile)

        plt.tight_layout()
        plt.savefig(f"{output_prefix}_all_methods.png", dpi=300)
        plt.close()

        # --- 3D and distance visualization ---
        figure_output_prefix = f"{fig_prefix}_{tract_name}"
        work_output_prefix = f"{work_prefix}_{tract_name}"
        plot_3d_refs(streamlines, ref_curves, work_output_prefix, max_display=300)
        plot_distance_histograms(streamlines, ref_curves, work_output_prefix, no_legend=args.no_legend)

        all_profiles[tract_name] = profiles
        all_ref_curves[tract_name] = ref_curves

    # ------------------------------------------------------------
    # === Global overlay plot across tracts ===
    plt.figure(figsize=profile_figsize)
    ax = plt.gca()

    # determine x-axis once
    n_nodes = None

    for t_idx, (tract_name, profs) in enumerate(all_profiles.items()):
        color = tract_colors[t_idx]
        if args.average_centroid:
            key = "average"
        else:
            key = list(profs.keys())[0]

        n_nodes = profs[key].shape[-1]
        x = np.linspace(0, 100, n_nodes)   # normalized tract position in %

        plt.plot(x, profs[key], color=color, lw=2.5, label=tract_labels[t_idx])

        disp_key = f"{key}_dispersion"
        if disp_key in profs and profs[disp_key] is not None and not args.no_dispersion:
            disp = profs[disp_key]
            plt.fill_between(
                x,
                profs[key] - disp,
                profs[key] + disp,
                color=color,
                alpha=0.25,
                linewidth=0,
            )

    plt.xlabel("Position along tract (%)", fontsize=fontsize)
    plt.ylabel(args.ylabel, fontsize=fontsize)

    if ymin is not None and ymax is not None:
        plt.ylim(ymin, ymax)

    # explicit x-axis control
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.set_xticklabels(["0", "50", "100"])


    # plt.xlabel("Node", fontsize=20)
    # ax.set_xlim(0, n_nodes - 1)
    # ax.set_xticks([0, n_nodes // 2, n_nodes - 1])
    # ax.set_xticklabels(["0", f"{n_nodes//2}", f"{n_nodes}"])

    nature_style_plot(ax, ymin, ymax, fontsize=fontsize,
                      add_origin_padding=True, 
                      pad_fraction=0.02,
                      n_yticks=3, y_decimals=2)

    #plt.title("Multi-Tract Profile Comparison")
    if not args.no_legend:
        plt.legend(fontsize=fontsize)

    plt.tight_layout()
    plt.savefig(f"{fig_prefix}_multi_tracts.png", dpi=300)
    plt.close()
    print(f"[INFO] Global overlay saved → {args.output}_multi_tracts.png")

    if args.boxplot_summary:
        boxplot_profiles = {}

        for t_idx, (tract_name, profs) in enumerate(all_profiles.items()):
            if args.average_centroid and "average" in profs:
                key = "average"
            else:
                key = list(profs.keys())[0]

            if key.endswith("_dispersion"):
                continue

            profile_vals = np.asarray(profs[key]).squeeze()

            if args.tract_labels:
                label = tract_labels[t_idx]
            else:
                label = tract_name

            if args.boxplot_stat == "profile":
                boxplot_profiles[label] = profile_vals
            elif args.boxplot_stat == "mean":
                boxplot_profiles[label] = np.array([np.mean(profile_vals)])

        ordered_labels = list(boxplot_profiles.keys())
        boxplot_colors = [tract_colors[i % len(tract_colors)] for i in range(len(ordered_labels))]

        plot_profile_boxplot(
            boxplot_profiles,
            ordered_labels,
            f"{args.output}_profile_boxplot.png",
            ylabel=args.ylabel,
            title=args.boxplot_title,
            xlabel=args.boxplot_x_label,
            colors=boxplot_colors,
            no_legend=args.no_legend,
            ymin=boxplot_ymin, 
            ymax=boxplot_ymax,
            boxplot_width=args.boxplot_width,
            boxplot_spacing=args.boxplot_spacing,
            boxplot_err=args.boxplot_err,
            fontsize=args.boxplot_font_size,
            figsize=boxplot_figsize
            
        )
    if len(all_profiles) >= 2:            
       # === Pairwise comparison between tract profiles ===
       comparison_rows = []
   
       for (tract_name_1, profs1), (tract_name_2, profs2) in combinations(all_profiles.items(), 2):
           if args.average_centroid:
               key1 = "average"
               key2 = "average"
           else:
               key1 = list(profs1.keys())[0]
               key2 = list(profs2.keys())[0]
   
           profile1 = profs1[key1]
           profile2 = profs2[key2]
   
           metrics = compare_profiles(profile1, profile2)
   
           row = {
               "tract_1": tract_name_1,
               "tract_2": tract_name_2,
               "method_1": key1,
               "method_2": key2,
               **metrics
           }
           comparison_rows.append(row)
   
       # Save as CSV
       csv_path = f"{args.output}_profile_comparisons.csv"
       with open(csv_path, "w", newline="") as f:
           writer = csv.DictWriter(f, fieldnames=comparison_rows[0].keys())
           writer.writeheader()
           writer.writerows(comparison_rows)
   
       print(f"[INFO] Saved pairwise profile comparisons → {csv_path}")
   
       plot_profile_similarity_matrix(
           all_profiles,
           tract_labels,
           f"{args.output}_profile_similarity_matrix.png",
           use_key="average" if args.average_centroid else "first",
           metric="pearson"
       )
       print(f"[INFO] Saved similarity matrix → {args.output}_profile_similarity_matrix.png")
    if True:    
      base_dir = os.path.dirname(args.output)
      if base_dir == "":
          base_dir = "."
      
      tract_profiles_dir = os.path.join(base_dir, "tract_profiles")
      os.makedirs(tract_profiles_dir, exist_ok=True)
      
      csv_path = os.path.join(tractmeasures_dir, "tractmeasures.csv")
      
      rows = list(tract_measure_map.values())
      rows.sort(key=lambda r: (r["structureID"], r["nodeID"]))
      
      metric_names = sorted({
          k[:-3] if k.endswith("_sd") else k
          for row in rows
          for k in row.keys()
          if k not in {"subjectID","structureID","nodeID","x_coords","y_coords","z_coords"}
          and not k.endswith("_coords")
      })
      
      fieldnames = ["subjectID","structureID","nodeID"]
      
      for metric in metric_names:
          fieldnames.extend([metric,f"{metric}_sd"])
      
      fieldnames.extend(["x_coords","y_coords","z_coords"])
      
      with open(csv_path,"w",newline="") as f:
          writer = csv.DictWriter(f,fieldnames=fieldnames)
          writer.writeheader()
          writer.writerows(rows)
      
      print(f"[INFO] Saved tract measures → {csv_path}")
    return




if __name__ == "__main__":
    try:
        import scipy  # noqa: F401
    except Exception as e:
        print("[WARN] SciPy import issue; profiles may still work, but fix your env if possible.")
        print(e)
    main()
