#!/usr/bin/env python3
"""Compute Dasgupta's cost for ATT tree structures.

Dasgupta's cost: sum over all pairs (i,j) of similarity(i,j) * |leaves(LCA(i,j))|
Lower cost = better hierarchical clustering (similar nodes meet in smaller subtrees).

Usage:
    python compute_dasgupta.py Results/bm_*D5*.pickle Results/bm_*D10*.pickle
    python compute_dasgupta.py --all-dims  # auto-find D1, D5, D10 for subject 100307
"""
import sys, pickle, re, argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

def load_tree(pkl_path):
    """Load tree structure from pickle. Returns (nleaf, nedge, node_connection)."""
    with open(pkl_path, 'rb') as f:
        bm = pickle.load(f)
    return bm.nleaf, bm.nedge, bm.node_connection

def build_adjacency(nleaf, nedge, node_connection):
    """Build adjacency list from node_connection."""
    adj = defaultdict(list)
    for j in range(len(node_connection)):
        nc = node_connection[j]
        parent, child1, child2 = nc[0], nc[1], nc[2]
        adj[parent].append(child1)
        adj[parent].append(child2)
        adj[child1].append(parent)
        adj[child2].append(parent)
    return adj

def count_leaves_under(node, adj, nleaf, parent=-1):
    """Count number of leaves in subtree rooted at node (coming from parent)."""
    if node < nleaf:
        return 1
    count = 0
    for neighbor in adj[node]:
        if neighbor != parent:
            count += count_leaves_under(neighbor, adj, nleaf, node)
    return count

def find_lca_leaves(nleaf, nedge, node_connection):
    """For each pair of leaves, find |leaves under LCA|.

    Uses BFS from each leaf to root to get ancestor paths,
    then LCA = first common ancestor.
    Returns: nleaf x nleaf matrix of |leaves under LCA(i,j)|.
    """
    adj = build_adjacency(nleaf, nedge, node_connection)

    # Find root: internal node with only 2 neighbors (top of tree)
    # Actually, find any internal node and use tree structure
    # Use Euler tour / DFS approach for efficient LCA

    # Pick root as the first internal node's parent edge
    root = node_connection[0][0]
    # Walk up to find actual root
    visited = set()
    cur = root
    while True:
        visited.add(cur)
        parents = [n for n in adj[cur] if n not in visited and n >= nleaf]
        if not parents:
            break
        cur = parents[0]
    root = cur

    # BFS from root to assign depths and parents
    depth = np.zeros(nedge, dtype=int)
    parent = np.full(nedge, -1, dtype=int)
    subtree_leaves = np.zeros(nedge, dtype=int)

    # DFS to compute depth, parent, subtree_leaves
    stack = [(root, -1, False)]
    order = []
    while stack:
        node, par, processed = stack.pop()
        if processed:
            # Post-order: compute subtree_leaves
            if node < nleaf:
                subtree_leaves[node] = 1
            else:
                subtree_leaves[node] = sum(
                    subtree_leaves[ch] for ch in adj[node] if ch != par
                )
            continue
        depth[node] = depth[par] + 1 if par >= 0 else 0
        parent[node] = par
        order.append(node)
        stack.append((node, par, True))  # post-order marker
        for ch in adj[node]:
            if ch != par:
                stack.append((ch, node, False))

    # Binary lifting for LCA (log N levels)
    LOG = max(1, int(np.log2(nedge)) + 1)
    up = np.full((nedge, LOG), -1, dtype=int)
    up[:, 0] = parent
    for k in range(1, LOG):
        for v in range(nedge):
            if up[v, k-1] >= 0:
                up[v, k] = up[up[v, k-1], k-1]

    def lca(u, v):
        if depth[u] < depth[v]:
            u, v = v, u
        diff = depth[u] - depth[v]
        for k in range(LOG):
            if (diff >> k) & 1:
                u = up[u, k]
        if u == v:
            return u
        for k in range(LOG - 1, -1, -1):
            if up[u, k] != up[v, k]:
                u = up[u, k]
                v = up[v, k]
        return up[u, 0]

    # Compute LCA leaves matrix for all leaf pairs
    lca_size = np.zeros((nleaf, nleaf), dtype=int)
    for i in range(nleaf):
        for j in range(i + 1, nleaf):
            l = lca(i, j)
            lca_size[i, j] = subtree_leaves[l]
            lca_size[j, i] = subtree_leaves[l]

    return lca_size

def load_similarity(data_dir, data_name):
    """Load training data and compute pairwise correlation as similarity."""
    dat_path = Path(data_dir) / f"{data_name}_sample.dat"
    des_path = Path(data_dir) / f"{data_name}.des"

    # Read descriptor
    with open(des_path) as f:
        lines = f.readlines()
    size = int(lines[0].split('=')[1])

    # Read one-hot data
    data = np.fromfile(str(dat_path), dtype=np.float32)
    q = 2  # binary
    n_vars = size
    n_samples = len(data) // (n_vars * q)
    data = data.reshape(n_samples, n_vars, q)

    # Convert one-hot to binary (take channel 1)
    binary = data[:, :, 1]  # (n_samples, n_vars)

    # Pairwise correlation (absolute value for similarity)
    corr = np.corrcoef(binary.T)  # (n_vars, n_vars)
    np.fill_diagonal(corr, 0)
    similarity = np.abs(corr)

    return similarity

def load_network_similarity(parc_path="Data/parcellation/Schaefer2018_200Parcels_7Networks_order.dlabel.nii"):
    """Build similarity matrix: 1 if same network, 0 if different."""
    import nibabel as nib
    parc = nib.load(str(parc_path))
    label_axis = parc.header.get_axis(0)
    label_map = label_axis.label[0]

    networks = []
    for i in range(1, 201):
        name = label_map[i][0]  # e.g. '7Networks_LH_Vis_1'
        for net in ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']:
            if f'_{net}_' in name:
                networks.append(net)
                break
        else:
            networks.append('Unknown')

    n = len(networks)
    similarity = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and networks[i] == networks[j]:
                similarity[i, j] = 1.0
    return similarity

def dasgupta_cost(lca_size, similarity):
    """Compute Dasgupta's cost = sum_{i<j} similarity(i,j) * |leaves(LCA(i,j))|."""
    n = lca_size.shape[0]
    cost = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            cost += similarity[i, j] * lca_size[i, j]
    return cost

def dasgupta_cost_normalized(lca_size, similarity):
    """Normalized Dasgupta's cost: divide by sum of similarities and N."""
    n = lca_size.shape[0]
    total_sim = 0.0
    cost = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            cost += similarity[i, j] * lca_size[i, j]
            total_sim += similarity[i, j]
    if total_sim == 0:
        return 0
    return cost / total_sim

def random_tree_baseline(nleaf, similarity, n_trials=100):
    """Compute Dasgupta's cost for random binary trees (baseline)."""
    costs = []
    for trial in range(n_trials):
        # Random binary tree: repeatedly merge random pairs
        leaves = list(range(nleaf))
        np.random.shuffle(leaves)

        # Build random tree
        next_id = nleaf
        subtree_size = {}
        for i in range(nleaf):
            subtree_size[i] = 1

        parent_map = {}
        active = list(range(nleaf))

        while len(active) > 1:
            # Pick two random elements (not necessarily adjacent) to merge
            idx1 = np.random.randint(0, len(active))
            idx2 = np.random.randint(0, len(active) - 1)
            if idx2 >= idx1:
                idx2 += 1
            a, b = active[idx1], active[idx2]

            new_node = next_id
            next_id += 1
            subtree_size[new_node] = subtree_size[a] + subtree_size[b]
            parent_map[a] = new_node
            parent_map[b] = new_node

            # Remove both, add new
            active = [x for x in active if x != a and x != b]
            active.append(new_node)

        # Compute LCA sizes for this random tree
        def find_ancestors(node):
            path = [node]
            while node in parent_map:
                node = parent_map[node]
                path.append(node)
            return path

        cost = 0.0
        for i in range(nleaf):
            anc_i = find_ancestors(i)
            anc_i_set = set(anc_i)
            for j in range(i + 1, nleaf):
                anc_j = find_ancestors(j)
                for a in anc_j:
                    if a in anc_i_set:
                        lca_node = a
                        break
                cost += similarity[i, j] * subtree_size[lca_node]
        costs.append(cost)

    return np.mean(costs), np.std(costs)

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('pickles', nargs='*', help='Pickle files to analyze')
    parser.add_argument('--all-dims', action='store_true',
                        help='Auto-find D1, D5, D10 for subject 100307')
    parser.add_argument('--data-dir', default='AdaptiveTensorTree/att_examples/BrainRest',
                        help='Directory containing training data')
    parser.add_argument('--data-name', default='brain_rest_100307_T1.0_S8',
                        help='Data name prefix')
    parser.add_argument('--random-trials', type=int, default=100,
                        help='Number of random tree trials for baseline')
    parser.add_argument('--network-labels', action='store_true',
                        help='Use network label similarity (1=same, 0=diff) instead of correlation')
    args = parser.parse_args()

    if args.all_dims:
        pkl_files = []
        for d in [1, 2, 5, 10]:
            p = Path(f'Results/bm_{args.data_name}_TY5_ST3_ALG0_ALP0.01_MAX10_D{d}_NS300_SE42_XS42.pickle')
            if p.exists():
                pkl_files.append(str(p))
        if not pkl_files:
            print("No pickle files found with --all-dims")
            sys.exit(1)
    else:
        pkl_files = args.pickles

    if not pkl_files:
        parser.print_help()
        sys.exit(1)

    # Load similarity matrix (same for all DIMs)
    if args.network_labels:
        print("Loading network label similarity (same network=1, different=0)...")
        similarity = load_network_similarity()
    else:
        print(f"Loading correlation similarity from {args.data_name}...")
        similarity = load_similarity(args.data_dir, args.data_name)
    nleaf = similarity.shape[0]
    print(f"  {nleaf} ROIs, mean similarity = {similarity.mean():.4f}")

    # Random baseline
    print(f"Computing random tree baseline ({args.random_trials} trials)...")
    rand_mean, rand_std = random_tree_baseline(nleaf, similarity, args.random_trials)
    print(f"  Random baseline: {rand_mean:.1f} +/- {rand_std:.1f}")
    rand_norm = rand_mean / similarity.sum() * 2  # normalized
    print(f"  Random normalized: {rand_norm:.2f}")

    # Compute for each pickle
    print()
    results = []
    for pkl_path in pkl_files:
        m = re.search(r'_D(\d+)_', pkl_path)
        dim = m.group(1) if m else '?'

        nleaf, nedge, node_connection = load_tree(pkl_path)
        lca_size = find_lca_leaves(nleaf, nedge, node_connection)
        cost = dasgupta_cost(lca_size, similarity)
        cost_norm = dasgupta_cost_normalized(lca_size, similarity)

        ratio = cost / rand_mean
        results.append((dim, cost, cost_norm, ratio))
        print(f"DIM={dim}: Dasgupta cost = {cost:.1f}, normalized = {cost_norm:.2f}, "
              f"ratio to random = {ratio:.3f}")

    # Summary table
    print(f"\n{'DIM':>4} {'Cost':>12} {'Normalized':>12} {'vs Random':>12}")
    print("-" * 44)
    print(f"{'rand':>4} {rand_mean:>12.1f} {rand_norm:>12.2f} {'1.000':>12}")
    for dim, cost, cost_norm, ratio in results:
        print(f"{dim:>4} {cost:>12.1f} {cost_norm:>12.2f} {ratio:>12.3f}")

if __name__ == '__main__':
    main()
