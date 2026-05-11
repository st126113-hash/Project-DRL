# Shooter-v0 Multi-Budget Video Export Project

This package contains four all-in-one notebook variants for running the same Shooter-v0 DRL workflow under different training budgets.

## Notebook variants

| Notebook budget | Intended use |
|---:|---|
| 250,000 timesteps | Fast backup run and practical tuning evidence. |
| 500,000 timesteps | Medium backup run when 250k is too short. |
| 1,000,000 timesteps | Longer final-agent check without the extreme 3M runtime. |
| 3,000,000 timesteps | Full long-run experiment for maximum training evidence, expected to take much longer. |

## Method framing

The final method is **Hierarchical PPO with a deterministic Lead-Aim Controller**, not pure end-to-end PPO. Direct Dueling Double DQN, A2C, and raw PPO remain low-level DRL baselines. Action 6 is disabled for fair evaluation; the final low-level action sent to Shooter-v0 is always within actions 0–5.

## Video export

The notebook variants enable baseline, final Hierarchical PPO, and comparison video export flags. Videos require a working PyOpenGL/Pygame render path and available checkpoints or trained model objects.

## Artifact formats

- Dueling Double DQN uses native PyTorch `.pt` checkpoints.
- A2C, PPO, and Hierarchical PPO use Stable-Baselines3 `.zip` checkpoints for reliable reload.
- Additional `.pt` policy-state exports are generated for instructor inspection when supported.
- CSV logs are written to `training_logs/` with the active artifact tag, such as `train_250000` or `train_3000000`.


## v9 ARTIFACT_TAG video-export fix

This package defines `ARTIFACT_TAG` before the early baseline-video export cell. The previous multi-budget package could raise `NameError: name 'ARTIFACT_TAG' is not defined` when the Lead-Aim baseline video cell was executed before the comparison configuration cell.


## v10 Video Export Flag Fix

This package defines `EXPORT_COMPARISON_3D_VIDEOS` near the start of each notebook and adds fallback definitions inside the video diagnostic cell. This prevents the video diagnostics from failing when the diagnostic cell is run before the comparison-video export cell.
