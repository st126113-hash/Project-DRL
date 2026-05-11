# Shooter-v0 2M Training Run

This package contains the **2,000,000-timestep** version of the Shooter-v0 Deep Reinforcement Learning experiment.

The purpose of this run is to provide a strong long-training checkpoint that is more complete than the 1M run, while still being lighter than the full 3M experiment.

## Training budget

| Run version | Training timesteps | Intended use |
|---:|---:|---|
| 2M | 2,000,000 | Long-run experiment for stronger policy learning and final performance comparison. |

## Method framing

The final method is **Hierarchical PPO with a deterministic Lead-Aim Controller**.

This means the agent is not a pure end-to-end PPO agent. PPO is used as the reinforcement learning component, while the deterministic Lead-Aim Controller assists with aiming behavior. Direct Dueling Double DQN, A2C, and raw PPO are kept as lower-level DRL baselines for comparison.

For fair evaluation, **Action 6 is disabled**. The final low-level action sent to Shooter-v0 is always within actions **0–5**.

## Expected artifacts

The 2M run is expected to produce or use the following artifacts:

- Trained model checkpoints
- Evaluation results
- Training logs
- Optional 3D gameplay videos
- Comparison videos between baseline and final policy

Typical output folders include:

```text
checkpoints/
outputs/
sample_uploaded_results/
training_logs/
```

## Artifact tag

For this 2M run, the recommended artifact tag is:

```python
ARTIFACT_TAG = "train_2000000"
```

This tag helps separate 2M outputs from other runs such as 100k, 250k, 500k, 1M, and 3M.

## Video export

The notebook supports exporting baseline, final Hierarchical PPO, and comparison videos.

Video generation requires:

- A working PyOpenGL/Pygame render path
- Available trained checkpoints or trained model objects
- Enough runtime to finish the evaluation episode

If video export fails, first check that the required checkpoint exists and that the notebook was run from the correct project folder.

## Checkpoint formats

- Dueling Double DQN uses native PyTorch `.pt` checkpoints.
- A2C, PPO, and Hierarchical PPO use Stable-Baselines3 `.zip` checkpoints.
- Additional `.pt` policy-state exports may be generated for instructor inspection when supported.
- CSV logs are written to `training_logs/` using the active artifact tag, such as `train_2000000`.

## Notes

This README is specific to the **2M training package**. It should be used inside the folder for the 2,000,000-timestep run only.

The original multi-budget README listed several run sizes together. This version is simplified so the repository folder clearly represents the 2M experiment.
