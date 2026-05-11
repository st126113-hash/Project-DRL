# Shooter-v0 1M Training Run

This package contains the **1,000,000 timesteps** version of the Shooter-v0 Deep Reinforcement Learning workflow.

The goal of this run is to provide a longer and more reliable training experiment than the 100k, 250k, and 500k backup runs, while still being shorter than the full 3M long-run experiment.

## Training budget

| Item | Value |
|---|---:|
| Environment | Shooter-v0 |
| Training budget | 1,000,000 timesteps |
| Main method | Hierarchical PPO with deterministic Lead-Aim Controller |
| Artifact tag | `train_1000000` |
| Evaluation action range | Actions 0-5 only |
| Action 6 | Disabled for fair evaluation |

## Method framing

The final method is **Hierarchical PPO with a deterministic Lead-Aim Controller**.

This is not a pure end-to-end PPO agent. The hierarchical policy uses PPO for high-level decision making, while the deterministic Lead-Aim Controller helps convert decisions into stable low-level actions.

Direct Dueling Double DQN, A2C, and raw PPO are treated as low-level DRL baselines for comparison.

## Why this 1M run is useful

The 1M run is intended for a stronger final-agent check than the smaller backup runs.

Compared with 100k, 250k, and 500k, this budget gives the PPO-based method more time to learn stable behavior, improve survival, and generate better evaluation evidence. Compared with the 3M run, it is still more practical when runtime is limited.

## Expected artifacts

The run may generate outputs such as:

- model checkpoints
- training logs
- evaluation summaries
- baseline 3D video
- final Hierarchical PPO 3D video
- comparison 3D videos

Typical output folders include:

```text
checkpoints/
outputs/
sample_uploaded_results/
training_logs/
```

## Artifact formats

- Dueling Double DQN uses native PyTorch `.pt` checkpoints.
- A2C, PPO, and Hierarchical PPO use Stable-Baselines3 `.zip` checkpoints for reliable reload.
- Additional `.pt` policy-state exports may be generated for instructor inspection.
- CSV logs are written to `training_logs/` using the active artifact tag, such as `train_1000000`.

## Video export

The notebook enables video export for baseline, final Hierarchical PPO, and comparison runs.

Videos require:

- working PyOpenGL/Pygame rendering
- available trained checkpoints or trained model objects
- correct export flags enabled in the notebook

The package includes fixes for common video-export configuration issues, including the early `ARTIFACT_TAG` definition and `EXPORT_COMPARISON_3D_VIDEOS` fallback logic.

## Notes

This package is designed as a self-contained 1M experiment package for submission evidence, evaluation, and comparison against shorter and longer training budgets.
