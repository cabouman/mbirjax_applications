# Plan: simplify the Lilly 4D CLI (`Lilly_recon.py` + `test_script_4d.sh`)

Status: **implemented** (commit 8c4cfd4).

## Goal
Make the 4D MACE deliverable easy for Lilly to run for real reconstructions by
exposing only the parameters they actually decide, while keeping every other knob
overridable for advanced use.

## Design principle — three tiers of parameters
- **Lilly decides** → visible in the shell script (`DATA_PATH`, `--output_path`, `--downsampling`).
- **Data / geometry / algorithm property** → Python default, overridable via `--flag`, hidden
  from the shell command but noted in a comment (`--frames_per_rotation`, `--frame_overlap_factor`).
- **Algorithm tuning** → Python default, overridable, invisible (`--rho_mann`, prox settings,
  `--weight_type`, ...).

---

## Changes to `Lilly_recon.py`

### 1. Collapse downsampling to one number
Replace `--downsample_row` / `--downsample_column` with a single `--downsampling`
(int, default 1), matching the sibling `nsi/Lilly_recon.py`.

Before (`preprocessing` group):
```python
g.add_argument("--downsample_row", type=int, default=1, help="Detector row subsampling factor.")
g.add_argument("--downsample_column", type=int, default=1, help="Detector column subsampling factor.")
g.add_argument("--subsample_view_factor", type=int, default=1, help="View subsampling factor.")
g.add_argument("--sharpness", type=float, default=1.0, help="mbirjax sharpness parameter.")
```
After:
```python
g.add_argument("--downsampling", type=int, default=1,
               help="Subsampling factor for detector rows and channels.")
g.add_argument("--sharpness", type=float, default=1.0, help="mbirjax sharpness parameter.")
```

Asymmetric row/column downsampling is geometrically valid but is a footgun with no upside
for real recons; keep the capability internal, expose one number.

### 2. Remove `--subsample_view_factor`
Drop the flag; let `get_sino_and_model` use its mbirjax default (1). `--num_frames` remains
the quick-test lever (fewer frames, each at full angular sampling), which is a cleaner
tradeoff for a moving phantom than degrading every frame's angular sampling.

Reversible: if a full-view 4D recon turns out too slow for the compute budget, re-add it as a
Python-only flag (still out of the shell).

### 3. Keep but hide `--frames_per_rotation` and `--frame_overlap_factor`
No change to the CLI code — both stay as `argparse` args with defaults 6 / 2.0 and remain
overridable. They leave the shell command but are **both documented in a comment** in
`test_script_4d.sh` so an advanced user can find and override them.

- `frame_overlap_factor = 2.0` — sets each frame's angular span to
  `factor * (360 / frames_per_rotation)` degrees, so it fixes both the views per frame and
  the number of frames: on the Lilly phantom (2.5 deg/view, 2400 views) 2.0 gives a 120 deg
  48-view frame and 99 frames, while 4.0 gives 240 deg / 96 views and 97 frames.  It trades
  temporal resolution against per-frame SNR rather than being pure tuning, but 2.0 is the
  validated value, so it is still reasonable to bury as a default.
- `frames_per_rotation = 6` — encodes acquisition geometry (anchor points 60 deg apart ->
  period-6). It is a property of the data, not the algorithm, and cannot be auto-derived from
  the NSI metadata in the current interface, so a differently-gated scan must override it.

### 3b. Rename `--max_iterations` -> `--max_mace_iterations` (kept visible in the shell)
The number of outer MACE iterations is a meaningful quality/time knob for a real recon, so it
stays **visible in the shell**. Rename the flag for clarity: the sibling MAR script
(`nsi/Lilly_recon.py`) already has a `--max_iterations` that means *MBIR* iterations, so reusing
that name here for *MACE outer* iterations would be confusing. `--max_mace_iterations` names the
concept unambiguously across the two deliverables.

Before (`MACE algorithm` group):
```python
g.add_argument("--max_iterations", type=int, default=10, help="Maximum number of outer MACE iterations.")
```
After:
```python
g.add_argument("--max_mace_iterations", type=int, default=10,
               help="Maximum number of outer MACE iterations.")
```
Also update the two references in `main()`:
```python
recon_4d, recon_dict = mace_model.recon(
    sino, weights=weights,
    max_iterations=args.max_mace_iterations,          # was args.max_iterations
    ...
)
...
print(f"[INFO] Iterations run: {recon_dict['recon_params']['iterations completed']} "
      f"of {args.max_mace_iterations}.")               # was args.max_iterations
```
The mbirjax `recon(max_iterations=...)` keyword is unchanged; only the CLI flag name changes.

### 4. Update the call site in `main()`
Before:
```python
sino, ct_model = mjp.nsi.get_sino_and_model(
    dataset_dir,
    downsample_factor=[args.downsample_row, args.downsample_column],
    subsample_view_factor=args.subsample_view_factor,
    auto_crop=True,
)
```
After:
```python
sino, ct_model = mjp.nsi.get_sino_and_model(
    dataset_dir,
    downsample_factor=[args.downsampling, args.downsampling],
    auto_crop=True,
)
```

### 5. Update `append_run_info()`
Before:
```python
f.write(f"downsample (row, col) = ({args.downsample_row}, {args.downsample_column})\n")
f.write(f"subsample_view_factor = {args.subsample_view_factor}\n")
```
After:
```python
f.write(f"downsampling         = {args.downsampling}\n")
```

No other logic changes. All mbirjax calls stay as-is (verified against the
`4DCT_for_merging` branch of mbirjax).

---

## Changes to `test_script_4d.sh`
Trim to what Lilly actually sets; everything else is a documented override.

```bash
cd "$(dirname "${BASH_SOURCE[0]}")"

DATA_PATH=/depot/bouman/data/Lilly/4DCT/Phantom_30s_Run1_Dec2024/
OUTPUT_PATH=./output

mkdir -p "$OUTPUT_PATH"
mkdir -p ~/4dct_logs/

python Lilly_recon.py \
  --data_path           "$DATA_PATH" \
  --output_path         "$OUTPUT_PATH" \
  --downsampling        1 \
  --max_mace_iterations 10 \
  2>&1 | tee ~/4dct_logs/recon_4d_run.log

# Quick test - reconstruct only the first N time frames, add:
#   --num_frames 25 \
#
# Advanced (leave at defaults unless you know why):
#   --frames_per_rotation 6      # time frames per 360 deg; must match the gating geometry
#   --frame_overlap_factor 2.0   # frames sharing any given view (MACE tuning)
```

The leading `cd` makes the script independent of the caller's working directory, so
`bash /any/path/test_script_4d.sh` still finds `Lilly_recon.py` and writes `./output` beside
the script rather than wherever it was invoked from.

Removed from the visible script: `--frames_per_rotation`, `--frame_overlap_factor`,
`--subsample_view_factor`, `--weight_type`. Both frame flags are documented in the comment
above; all removed flags remain reachable via `--help`. `--max_mace_iterations` stays visible.

---

## Net effect
- CLI surface for downsampling: `--downsample_row`, `--downsample_column`,
  `--subsample_view_factor` (3 flags) -> `--downsampling` (1 flag).
- `--max_iterations` -> `--max_mace_iterations` (renamed for clarity vs. the MAR script).
- Visible shell knobs: 9 -> 4 (`DATA_PATH`, `--output_path`, `--downsampling`,
  `--max_mace_iterations`), with quick-test and advanced overrides shown as comments.

## Decisions
- `frames_per_rotation` and `frame_overlap_factor` are both kept **overridable but hidden**,
  documented in the shell comment. Neither is dropped from the CLI.
- `max_mace_iterations` is kept **visible** in the shell as a real quality/time knob.
