#!/usr/bin/env python3
"""
Preprocess HCP resting-state fMRI data for ATT analysis.

Pipeline:
  1. Load CIFTI .dtseries.nii
  2. Extract Schaefer 200 ROI mean timeseries (vertex-level matching)
  3. Z-score normalization per ROI
  4. Discretization: binary (q=2) or ternary (q=3: suppression/baseline/activation)
  5. Temporal subsampling (every 8 TR ≈ 6s, matching hemodynamic response)
  6. One-hot encode and save in ATT format (.dat + .des)

Usage:
  python prep_fmri_to_binary.py --subject 100307
  python prep_fmri_to_binary.py --subject all --theta 1.0 --subsample 8
  python prep_fmri_to_binary.py --subject all --q_state 3 --theta 1.0
"""

import argparse
import os
import struct
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

# ── Configuration ──────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "Data"
PARC_PATH = DATA_DIR / "parcellation" / "Schaefer2018_200Parcels_7Networks_order.dlabel.nii"
OUTPUT_DIR = Path(__file__).parent / "AdaptiveTensorTree" / "att_examples" / "BrainRest"

SUBJECTS = ["100307", "100408", "101107", "101309", "101915"]
RUNS = ["rfMRI_REST1_LR", "rfMRI_REST1_RL", "rfMRI_REST2_LR", "rfMRI_REST2_RL"]
N_ROIS = 200


def build_vertex_maps(fmri_bm, parc_bm):
    """Build mappings between fMRI and parcellation vertex indices."""
    # fMRI: (structure, vertex) -> fmri column index
    fmri_map = {}
    for name, slc, bm in fmri_bm.iter_structures():
        if "CORTEX" in name:
            for i, v in enumerate(bm.vertex):
                fmri_map[(name, v)] = slc.start + i

    # Parcellation: (structure, vertex) -> parcel label
    return fmri_map


def extract_roi_timeseries(fmri_path, parc_img, fmri_vertex_map, roi_fmri_indices):
    """Extract mean timeseries for each ROI from a single run."""
    img = nib.load(str(fmri_path))
    data = img.get_fdata()  # (T, 91282)

    # If roi_fmri_indices not yet built, build it
    if roi_fmri_indices is None:
        parc_data = parc_img.get_fdata().squeeze()
        parc_bm = parc_img.header.get_axis(1)
        roi_fmri_indices = {r: [] for r in range(1, N_ROIS + 1)}
        for name, slc, bm in parc_bm.iter_structures():
            if "CORTEX" in name:
                for i, v in enumerate(bm.vertex):
                    label = int(parc_data[slc.start + i])
                    key = (name, v)
                    if label > 0 and key in fmri_vertex_map:
                        roi_fmri_indices[label].append(fmri_vertex_map[key])

    # Compute mean timeseries per ROI
    roi_ts = np.zeros((data.shape[0], N_ROIS))
    for roi in range(1, N_ROIS + 1):
        idx = roi_fmri_indices[roi]
        if len(idx) > 0:
            roi_ts[:, roi - 1] = data[:, idx].mean(axis=1)

    return roi_ts, roi_fmri_indices


def preprocess(roi_ts, theta=1.0, subsample=8, q_state=2, theta_low=None, theta_high=None):
    """Z-score, discretize, and subsample."""
    if theta_low is None:
        theta_low = theta
    if theta_high is None:
        theta_high = theta

    # Z-score per ROI (column)
    mean = roi_ts.mean(axis=0, keepdims=True)
    std = roi_ts.std(axis=0, keepdims=True)
    z = (roi_ts - mean) / (std + 1e-10)

    if q_state == 2:
        # Binary: point process binarization
        discrete = (z > theta_high).astype(np.uint8)
    elif q_state == 3:
        # Ternary: suppression (0) / baseline (1) / activation (2)
        discrete = np.ones(z.shape, dtype=np.uint8)  # default: baseline
        discrete[z < -theta_low] = 0   # suppression
        discrete[z > theta_high] = 2   # activation
    else:
        raise ValueError(f"q_state must be 2 or 3, got {q_state}")

    # Temporal subsampling
    discrete = discrete[::subsample, :]

    return discrete


def save_att_format(data_train, data_test, output_dir, name, q_state=2):
    """Save in ATT format: .des descriptor + _sample.dat + _test.dat"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_train, n_vars = data_train.shape
    n_test = data_test.shape[0]

    # One-hot encode: (n_samples, n_vars) -> (n_samples, n_vars, q_state)
    onehot_train = np.zeros((n_train, n_vars, q_state), dtype=np.float32)
    onehot_test = np.zeros((n_test, n_vars, q_state), dtype=np.float32)
    for i in range(n_train):
        for j in range(n_vars):
            onehot_train[i, j, data_train[i, j]] = 1.0
    for i in range(n_test):
        for j in range(n_vars):
            onehot_test[i, j, data_test[i, j]] = 1.0

    # Save .dat files (flat binary, float32)
    sample_path = output_dir / f"{name}_sample.dat"
    test_path = output_dir / f"{name}_test.dat"
    onehot_train.tofile(str(sample_path))
    onehot_test.tofile(str(test_path))

    # Save .des file
    des_path = output_dir / f"{name}.des"
    with open(des_path, "w") as f:
        f.write(f"size={n_vars}\n")
        f.write(f"dtype=<f4\n")
        f.write(f"q={q_state}\n")

    print(f"  Saved: {sample_path.name} ({n_train} samples)")
    print(f"  Saved: {test_path.name} ({n_test} samples)")
    print(f"  Saved: {des_path.name}")


def process_subject(subject, parc_img, fmri_vertex_map, theta, subsample,
                    q_state=2, theta_low=None, theta_high=None):
    """Process one subject: load 4 runs, preprocess, split train/test, save."""
    print(f"\n{'='*60}")
    print(f"Processing subject {subject} (q={q_state})")
    print(f"{'='*60}")

    all_discrete = []
    run_labels = []  # track which run each sample came from
    roi_fmri_indices = None

    for run in RUNS:
        cifti_dir = DATA_DIR / subject / "MNINonLinear" / "Results" / run
        cifti_files = list(cifti_dir.glob("*_Atlas_MSMAll_hp2000_clean*_tclean.dtseries.nii"))
        if not cifti_files:
            print(f"  WARNING: No CIFTI file found for {run}, skipping")
            continue

        cifti_path = cifti_files[0]
        print(f"  Loading {run}... ", end="", flush=True)

        roi_ts, roi_fmri_indices = extract_roi_timeseries(
            cifti_path, parc_img, fmri_vertex_map, roi_fmri_indices
        )
        discrete = preprocess(roi_ts, theta=theta, subsample=subsample,
                              q_state=q_state, theta_low=theta_low, theta_high=theta_high)
        all_discrete.append(discrete)
        run_labels.extend([run] * discrete.shape[0])
        # Print state distribution
        if q_state == 2:
            print(f"{roi_ts.shape[0]} TRs -> {discrete.shape[0]} samples "
                  f"(active rate: {discrete.mean():.3f})")
        else:
            for s in range(q_state):
                rate = (discrete == s).mean()
                labels = {0: "suppress", 1: "baseline", 2: "active"}
                print(f"{labels.get(s, f'state{s}')}: {rate:.3f}  ", end="")
            print(f"({roi_ts.shape[0]} TRs -> {discrete.shape[0]} samples)")

    if not all_discrete:
        print(f"  ERROR: No data found for subject {subject}")
        return

    all_discrete = np.concatenate(all_discrete, axis=0)
    print(f"\n  Total: {all_discrete.shape[0]} samples x {all_discrete.shape[1]} ROIs")

    # Train/test split: runs 1&2 (LR+RL) = train, runs 3&4 = test
    n_run1 = len([r for r in run_labels if "REST1" in r])
    data_train = all_discrete[:n_run1]
    data_test = all_discrete[n_run1:]

    print(f"  Train (REST1): {data_train.shape[0]} samples")
    print(f"  Test  (REST2): {data_test.shape[0]} samples")

    # Save — append Q{q} only for q>2 to preserve backward compatibility
    name = f"brain_rest_{subject}_T{theta}_S{subsample}"
    if q_state > 2:
        name += f"_Q{q_state}"
    save_att_format(data_train, data_test, OUTPUT_DIR, name, q_state=q_state)


def main():
    parser = argparse.ArgumentParser(description="Preprocess HCP fMRI for ATT")
    parser.add_argument("--subject", default="100307",
                        help="Subject ID or 'all' for all subjects")
    parser.add_argument("--theta", type=float, default=1.0,
                        help="Binarization threshold (z-score, default: 1.0)")
    parser.add_argument("--subsample", type=int, default=8,
                        help="Temporal subsampling factor (default: 8)")
    parser.add_argument("--q_state", type=int, default=2,
                        help="Number of discrete states (2=binary, 3=ternary, default: 2)")
    parser.add_argument("--theta_low", type=float, default=None,
                        help="Lower threshold for suppression (default: same as --theta)")
    parser.add_argument("--theta_high", type=float, default=None,
                        help="Upper threshold for activation (default: same as --theta)")
    args = parser.parse_args()

    if args.theta_low is None:
        args.theta_low = args.theta
    if args.theta_high is None:
        args.theta_high = args.theta

    # Check parcellation
    if not PARC_PATH.exists():
        print(f"ERROR: Parcellation not found at {PARC_PATH}")
        sys.exit(1)

    print("Loading Schaefer 200 parcellation...")
    parc_img = nib.load(str(PARC_PATH))

    # Build vertex mapping (once)
    print("Building vertex mapping...")
    # Use first available subject to get fMRI brain model axis
    first_subj = args.subject if args.subject != "all" else SUBJECTS[0]
    first_run = RUNS[0]
    ref_path = (DATA_DIR / first_subj / "MNINonLinear" / "Results" / first_run)
    ref_cifti = list(ref_path.glob("*_Atlas_MSMAll_hp2000_clean*_tclean.dtseries.nii"))[0]
    ref_img = nib.load(str(ref_cifti))
    fmri_bm = ref_img.header.get_axis(1)

    fmri_vertex_map = {}
    for name, slc, bm in fmri_bm.iter_structures():
        if "CORTEX" in name:
            for i, v in enumerate(bm.vertex):
                fmri_vertex_map[(name, v)] = slc.start + i
    print(f"  Mapped {len(fmri_vertex_map)} cortical vertices")

    # Process subjects
    subjects = SUBJECTS if args.subject == "all" else [args.subject]
    for subj in subjects:
        process_subject(subj, parc_img, fmri_vertex_map, args.theta, args.subsample,
                        q_state=args.q_state, theta_low=args.theta_low,
                        theta_high=args.theta_high)

    print(f"\n{'='*60}")
    print(f"Done! Output directory: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
