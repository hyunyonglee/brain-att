# Q-State Upgrade Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend ATT pipeline from binary (q=2) to q-state discretization, starting with q=3 for fMRI data.

**Architecture:** Fork approach — copy born_machine.py to born_machine_qstate.py for sampling changes, modify prep_fmri_to_binary.py for q-state discretization, modify apply_att.py to read q from .des and reshape accordingly. Original born_machine.py stays untouched.

**Tech Stack:** Python, PyTorch, NumPy, nibabel (preprocessing only)

---

### Task 1: Modify prep_fmri_to_binary.py for q-state discretization

**Files:**
- Modify: `prep_fmri_to_binary.py`

**Step 1: Add q-state arguments to argparser**

In `main()`, add three new arguments after the existing `--subsample`:

```python
parser.add_argument("--q_state", type=int, default=2,
                    help="Number of discrete states (2=binary, 3=ternary, default: 2)")
parser.add_argument("--theta_low", type=float, default=None,
                    help="Lower threshold for suppression (default: same as --theta)")
parser.add_argument("--theta_high", type=float, default=None,
                    help="Upper threshold for activation (default: same as --theta)")
```

After `args = parser.parse_args()`, add defaults:
```python
if args.theta_low is None:
    args.theta_low = args.theta
if args.theta_high is None:
    args.theta_high = args.theta
```

**Step 2: Update global Q and pass q_state to functions**

Remove global `Q = 2`. Instead, pass `q_state` through the pipeline:
- `process_subject()`: add `q_state` parameter
- `save_att_format()`: add `q_state` parameter
- `preprocess()`: add `q_state`, `theta_low`, `theta_high` parameters

**Step 3: Update `preprocess()` for q-state discretization**

```python
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
        # Ternary: suppression / baseline / activation
        discrete = np.ones(z.shape, dtype=np.uint8)  # default: baseline (1)
        discrete[z < -theta_low] = 0   # suppression
        discrete[z > theta_high] = 2   # activation
    else:
        raise ValueError(f"q_state must be 2 or 3, got {q_state}")

    # Temporal subsampling
    discrete = discrete[::subsample, :]

    return discrete
```

**Step 4: Update `save_att_format()` to use q_state**

```python
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

    # Save .dat files
    onehot_train.tofile(str(output_dir / f"{name}_sample.dat"))
    onehot_test.tofile(str(output_dir / f"{name}_test.dat"))

    # Save .des file with q field
    with open(output_dir / f"{name}.des", "w") as f:
        f.write(f"size={n_vars}\n")
        f.write(f"dtype=<f4\n")
        f.write(f"q={q_state}\n")
```

**Step 5: Update `process_subject()` and naming**

- Pass `q_state`, `theta_low`, `theta_high` to `preprocess()`
- Pass `q_state` to `save_att_format()`
- Update naming: `name = f"brain_rest_{subject}_T{theta}_S{subsample}"` → append `_Q{q_state}` only when q_state > 2
- Print state distribution statistics

**Step 6: Commit**

```bash
git add prep_fmri_to_binary.py
git commit -m "feat: add q-state discretization support to preprocessing"
```

---

### Task 2: Modify apply_att.py to read q from .des

**Files:**
- Modify: `AdaptiveTensorTree/apply_att.py`

**Step 1: Read q from .des file**

After the `data_dtype` parsing block (line ~189), add:

```python
if "q" in data_description:
    q_state = int(data_description["q"])
else:
    q_state = 2  # backward compatible default
```

**Step 2: Replace hardcoded reshape `(-1, image_size, 2)` with `(-1, image_size, q_state)`**

Three locations in apply_att.py:
- Line 198: `sample_data = np.reshape(sample_data, (-1, image_size, q_state))`
- Line 212: `sample_data = np.reshape(sample_data, (-1, image_size, q_state))`
- Line 288: `test_data = np.reshape(test_data, (-1, image_size, q_state))`

**Step 3: Conditional import for q>2**

Replace line 30 `import born_machine` with:

```python
# born_machine import is deferred until q_state is known
```

After q_state is determined (after .des parsing), add:

```python
if q_state > 2:
    import born_machine_qstate as born_machine
else:
    import born_machine
```

**Step 4: Commit**

```bash
git add AdaptiveTensorTree/apply_att.py
git commit -m "feat: add q-state support to apply_att.py (read q from .des)"
```

---

### Task 3: Create born_machine_qstate.py

**Files:**
- Create: `AdaptiveTensorTree/born_machine_qstate.py` (copy of born_machine.py)

**Step 1: Copy born_machine.py**

```bash
cp AdaptiveTensorTree/born_machine.py AdaptiveTensorTree/born_machine_qstate.py
```

**Step 2: Modify `sampling_do()` method**

In `born_machine_qstate.py`, find the `sampling_do` method. Replace the binary-specific sampling logic (lines ~1556-1576):

Replace:
```python
        prob = new_weight[:, 1] / (new_weight[:, 0] + new_weight[:, 1])
        nsample = prob.shape[0]
        new_value = torch.empty(nsample, dtype=torch.uint8, device=self.base_device)
        prob[prob < 0] = 0.0
        prob[prob > 1] = 1.0
        if single_site_marginal_distribution:
            new_edge_matrix = torch.zeros((nsample, 2, 2), device=self.base_device)
            new_edge_matrix[:, 0, 0] = 1
            new_edge_matrix[:, 1, 1] = 1
            self.edge_matrix[ie] = new_edge_matrix
            self.probability_one[:, ie] = prob
        else:
            torch.bernoulli(prob, generator=self.rng, out=new_value)
            self.sampling_data[:, ie] = new_value
            new_edge_matrix = torch.zeros(nsample * 2 * 2, device=self.base_device)
            indices = (
                torch.arange(0, nsample * 2 * 2, 2 * 2, device=self.base_device)
                + new_value * 3
            )
            new_edge_matrix[indices] = 1.0
            self.edge_matrix[ie] = torch.reshape(new_edge_matrix, (nsample, 2, 2))
```

With q-state generalized version:
```python
        # Compute probabilities for each state
        q = new_weight.shape[1]
        nsample = new_weight.shape[0]
        # Normalize: p_i = w_i / sum(w)
        new_weight = torch.clamp(new_weight, min=0.0)
        total = new_weight.sum(dim=1, keepdim=True)
        probs = new_weight / (total + 1e-30)

        if single_site_marginal_distribution:
            new_edge_matrix = torch.zeros((nsample, q, q), device=self.base_device)
            for s in range(q):
                new_edge_matrix[:, s, s] = 1
            self.edge_matrix[ie] = new_edge_matrix
            if q == 2:
                self.probability_one[:, ie] = probs[:, 1]
            # For q>2, probability_one is not applicable
        else:
            new_value = torch.multinomial(probs, 1, generator=self.rng).squeeze(1).to(torch.uint8)
            self.sampling_data[:, ie] = new_value
            new_edge_matrix = torch.zeros((nsample, q, q), device=self.base_device)
            for i in range(nsample):
                new_edge_matrix[i, new_value[i], new_value[i]] = 1.0
            self.edge_matrix[ie] = new_edge_matrix
```

**Step 3: Commit**

```bash
git add AdaptiveTensorTree/born_machine_qstate.py
git commit -m "feat: create born_machine_qstate.py with q-state sampling support"
```

---

### Task 4: Test with existing q=2 data (backward compatibility)

**Step 1: Verify q=2 still works with modified apply_att.py**

```bash
cd brain-att
source venv/bin/activate
PYTHONPATH=AdaptiveTensorTree:$PYTHONPATH python AdaptiveTensorTree/apply_att.py brain_rest_100307_T1.0_S8 5 3 0 0.01 10 5 300 42 42 -N 100 --data_dir AdaptiveTensorTree/att_examples/BrainRest
```

Expected: runs without error, q=2 default kicks in, same behavior as before.

**Step 2: Commit if tests pass**

```bash
git commit --allow-empty -m "test: verify backward compatibility with q=2 data"
```

---

### Task 5: Generate q=3 data and test

**Step 1: Generate q=3 data for one subject**

Note: This requires HCP CIFTI data in Data/ directory. If not available on current machine, skip to Step 3.

```bash
python prep_fmri_to_binary.py --subject 100307 --q_state 3 --theta 1.0
```

Expected output: `brain_rest_100307_T1.0_S8_Q3_{sample,test}.dat` and `.des` with `q=3`.

**Step 2: Verify .des file contains q=3**

```bash
cat AdaptiveTensorTree/att_examples/BrainRest/brain_rest_100307_T1.0_S8_Q3.des
```

Expected:
```
size=200
dtype=<f4
q=3
```

**Step 3: Run ATT with q=3 data (smoke test)**

```bash
PYTHONPATH=AdaptiveTensorTree:$PYTHONPATH python AdaptiveTensorTree/apply_att.py brain_rest_100307_T1.0_S8_Q3 5 3 0 0.01 10 5 300 42 42 -N 100 --data_dir AdaptiveTensorTree/att_examples/BrainRest
```

Expected: runs without error, imports born_machine_qstate, reshape uses q=3.

**Step 4: Commit and push**

```bash
git add -A
git commit -m "feat: q-state upgrade complete (q=2 backward compatible, q=3 ready)"
git push
```

---

### Task Summary

| Task | Description | Files |
|------|------------|-------|
| 1 | Preprocessing q-state discretization | prep_fmri_to_binary.py |
| 2 | apply_att.py q support (.des → reshape) | AdaptiveTensorTree/apply_att.py |
| 3 | Fork born_machine for q-state sampling | AdaptiveTensorTree/born_machine_qstate.py |
| 4 | Backward compatibility test (q=2) | — |
| 5 | Generate q=3 data and smoke test | — |
