#!/usr/bin/env python3
"""Export ATT tree structure to CSV files for MATLAB visualization.

Usage:
    python export_tree_csv.py Results/bm_*.pickle
    python export_tree_csv.py Results/bm_*D5*.pickle Results/bm_*D10*.pickle
"""
import sys, csv, pickle, re
import numpy as np
import nibabel as nib
from pathlib import Path

PARC_PATH = Path("Data/parcellation/Schaefer2018_200Parcels_7Networks_order.dlabel.nii")

# Yeo 7 network colors (RGB)
NETWORK_COLORS = {
    'Vis': (120, 18, 134),
    'SomMot': (70, 130, 180),
    'DorsAttn': (0, 118, 14),
    'SalVentAttn': (196, 58, 250),
    'Limbic': (220, 248, 164),
    'Cont': (230, 148, 34),
    'Default': (205, 62, 78),
}

def load_roi_info(parc_path):
    """Load ROI names and network labels from Schaefer parcellation."""
    parc = nib.load(str(parc_path))
    label_axis = parc.header.get_axis(0)
    label_map = label_axis.label[0]

    roi_info = []
    for i in range(1, 201):
        full_name = label_map[i][0]  # e.g. '7Networks_LH_Vis_1'
        short_name = full_name.replace('7Networks_', '')  # 'LH_Vis_1'
        hemi = 'LH' if '_LH_' in full_name else 'RH'
        network = 'Unknown'
        for net in NETWORK_COLORS:
            if f'_{net}_' in full_name:
                network = net
                break
        r, g, b = NETWORK_COLORS.get(network, (128, 128, 128))
        roi_info.append({
            'name': short_name, 'network': network,
            'hemisphere': hemi, 'r': r, 'g': g, 'b': b,
        })
    return roi_info

def export_tree(pkl_path, roi_info, out_dir=None):
    """Export one pickle to tree_nodes_DIM{d}.csv and tree_edges_DIM{d}.csv."""
    with open(pkl_path, 'rb') as f:
        bm = pickle.load(f)

    # Extract DIM and data prefix from filename
    m = re.search(r'_D(\d+)_', str(pkl_path))
    dim = m.group(1) if m else 'X'
    fname = Path(pkl_path).stem
    prefix = 'group5_' if 'group5' in fname else ''

    if out_dir is None:
        out_dir = Path(pkl_path).parent
    out_dir = Path(out_dir)

    nleaf, nedge = bm.nleaf, bm.nedge

    # Nodes
    with open(out_dir / f'tree_nodes_{prefix}DIM{dim}.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['node_id', 'is_leaf', 'roi_index', 'roi_name',
                     'network', 'hemisphere', 'color_r', 'color_g', 'color_b'])
        for i in range(nedge):
            if i < nleaf:
                info = roi_info[i]
                w.writerow([i+1, 1, i+1, info['name'], info['network'],
                            info['hemisphere'], info['r'], info['g'], info['b']])
            else:
                w.writerow([i+1, 0, '', '', '', '', 128, 128, 128])

    # Edges (from node_connection: [parent_edge, child1, child2])
    edges = set()
    for j in range(len(bm.node_connection)):
        nc = bm.node_connection[j]
        edges.add((nc[1]+1, nc[0]+1))
        edges.add((nc[2]+1, nc[0]+1))

    with open(out_dir / f'tree_edges_{prefix}DIM{dim}.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['source', 'target'])
        for s, t in sorted(edges):
            w.writerow([s, t])

    print(f"DIM={dim}: exported (nleaf={nleaf}, edges={len(edges)}, counter={bm.counter})")

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    roi_info = load_roi_info(PARC_PATH)

    for pkl_path in sys.argv[1:]:
        try:
            export_tree(pkl_path, roi_info)
        except Exception as e:
            print(f"Error processing {pkl_path}: {e}")

if __name__ == '__main__':
    main()
