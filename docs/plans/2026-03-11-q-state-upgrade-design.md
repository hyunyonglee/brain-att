# Q-State Upgrade Design

## Goal
Extend ATT pipeline from binary (q=2) to q-state (default q=3) discretization of fMRI data.
3-state: suppression (z < -θ) / baseline (-θ ≤ z ≤ θ) / activation (z > θ)

## Approach: Fork (Method C)
- Copy born_machine.py → born_machine_qstate.py, modify sampling for q>2
- Keep original born_machine.py untouched
- Modify prep_fmri_to_binary.py and apply_att.py to support q parameter

## File Changes

### 1. prep_fmri_to_binary.py
- Add args: `--q_state` (default 2), `--theta_low` (default 1.0), `--theta_high` (default 1.0)
- 3-state discretization: z < -θ_low → 0, -θ_low ≤ z ≤ θ_high → 1, z > θ_high → 2
- One-hot: (n_samples, n_vars, q)
- .des file: add `q=<value>` field
- Output naming: `brain_rest_{subj}_T{theta}_S{subsample}_Q{q}`

### 2. apply_att.py
- Read `q` from .des file (default q=2 for backward compatibility)
- Change reshape from `(-1, image_size, 2)` to `(-1, image_size, q)`
- Import born_machine_qstate instead of born_machine when q>2

### 3. born_machine_qstate.py (new, forked from born_machine.py)
- sampling_do(): torch.bernoulli → torch.multinomial for q>2
- Edge matrix: (nsample, 2, 2) → (nsample, q, q)
- Store q as instance attribute

### 4. .des file format
- Add `q=3` field (backward compatible: missing q defaults to 2)

## Parameters
- θ symmetric by default (θ_low = θ_high = 1.0)
- At θ=1.0: activation ~16%, suppression ~16%, baseline ~68%
- Subsampling rate unchanged (8 TR)

## Data Naming Convention
- q=2: `brain_rest_100307_T1.0_S8` (existing, no change)
- q=3: `brain_rest_100307_T1.0_S8_Q3`

## Backward Compatibility
- q=2 data and code work exactly as before
- Original born_machine.py preserved
