# Shooter-v0 500k Training Run

This repository package contains a **500,000-timestep Shooter-v0 DRL experiment**. It is intended as a medium-budget run for stronger evidence than the 250k backup version, while still being shorter and easier to run than the 1M or 3M long-run experiments.

## Project summary

The project trains and evaluates agents for the custom **Shooter-v0** environment. The goal is to make the hunter survive longer, clear waves, and achieve a higher score while keeping the evaluation fair.

This 500k version is suitable for:

- medium-length training evidence,
- checking whether performance improves beyond the 250k run,
- producing more reliable comparison results,
- exporting baseline, final-agent, and comparison videos,
- showing a practical balance between runtime and result quality.

## Training budget

| Version | Timesteps | Intended use |
|---|---:|---|
| 500k run | 500,000 | Medium backup run when 250k is too short. |

Recommended artifact tag:

```python
ARTIFACT_TAG = "train_500000"
```

## Method framing

The final method is **Hierarchical PPO with a deterministic Lead-Aim Controller**.

This is not pure end-to-end PPO. The high-level policy is trained with PPO, while the deterministic Lead-Aim Controller helps with aiming and low-level action selection.

Baseline methods may include:

- Dueling Double DQN,
- A2C,
- raw PPO,
- Lead-Aim baseline.

For fair evaluation, **Action 6 is disabled**. The final low-level action sent to Shooter-v0 should always be within actions **0–5**.

## Expected outputs

The 500k run may generate outputs such as:

```text
checkpoints/
outputs/
training_logs/
sample_uploaded_results/
```

Common artifacts include:

- trained model checkpoints,
- evaluation CSV logs,
- baseline videos,
- final-agent videos,
- comparison videos,
- summary result files.

## Artifact formats

- Dueling Double DQN uses PyTorch `.pt` checkpoints.
- A2C, PPO, and Hierarchical PPO use Stable-Baselines3 `.zip` checkpoints.
- Extra `.pt` policy-state exports may be created for instructor inspection.
- CSV logs are written to `training_logs/` using the artifact tag, for example:

```text
training_logs/train_500000_*.csv
```

## Video export

This package supports video export for:

- Lead-Aim baseline,
- final Hierarchical PPO agent,
- comparison runs.

Video export requires a working PyOpenGL/Pygame render path. If videos are not created, check that:

- the render dependencies are installed,
- the checkpoint/model exists,
- the output folder exists,
- the video export flag is enabled.

Example video naming pattern:

```text
outputs/lead_aim_no_action6_real_3d_train_500000.mp4
outputs/final_hierarchical_ppo_real_3d_train_500000.mp4
```

## Important fixes included

### ARTIFACT_TAG fix

`ARTIFACT_TAG` should be defined before any video export cell. This prevents errors such as:

```text
NameError: name 'ARTIFACT_TAG' is not defined
```

### Comparison video flag fix

`EXPORT_COMPARISON_3D_VIDEOS` should be defined near the start of the notebook. Fallback definitions may also be included inside the video diagnostic cell to prevent failures when cells are run out of order.

## Suggested run order

1. Install requirements.
2. Import libraries.
3. Define global configuration.
4. Set `ARTIFACT_TAG = "train_500000"`.
5. Train or load baseline models.
6. Train or load Hierarchical PPO.
7. Evaluate the models.
8. Export videos.
9. Save logs and final artifacts.

## Notes

This 500k version is a medium-budget experiment. It should provide stronger evidence than the 250k run, but it is still more practical than 1M or 3M timesteps. It is useful for showing model improvement, creating comparison videos, and demonstrating the full Shooter-v0 DRL workflow without requiring the longest training time.
