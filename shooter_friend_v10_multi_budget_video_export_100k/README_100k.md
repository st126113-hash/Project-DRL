# Shooter-v0 100k Training Run

This repository contains the **100,000 timestep** Shooter-v0 DRL experiment package.  
It is intended as a fast, lightweight training run for testing the workflow, checking video export, and producing basic training evidence.

## Experiment budget

| Training budget | Intended use |
|---:|---|
| 100,000 timesteps | Fast test run / early experiment evidence / quick video-export check. |

This 100k version is not expected to outperform longer runs such as 250k, 500k, 1M, or 3M timesteps.  
Its main purpose is to confirm that the environment, training loop, checkpoints, logs, and video export pipeline work correctly.

## Method framing

The final method is **Hierarchical PPO with a deterministic Lead-Aim Controller**.

The project compares the final method against lower-level DRL baselines such as:

- Dueling Double DQN
- A2C
- Raw PPO

For fair evaluation, **Action 6 is disabled** in the final low-level policy.  
The final action sent to the Shooter-v0 environment is always within actions **0–5**.

## Video export

The notebook supports video export for:

- Lead-Aim baseline video
- Final Hierarchical PPO video
- Comparison videos

Video export requires:

- Working PyOpenGL and Pygame rendering
- Available trained model object or saved checkpoint
- Correct output folder path

Videos are saved under the `outputs/` folder when the export flags are enabled.

Example artifact tag for this version:

```python
ARTIFACT_TAG = "train_100000"
```

Example output video name:

```text
lead_aim_no_action6_real_3d_train_100000.mp4
```

## Artifact formats

- Dueling Double DQN uses native PyTorch `.pt` checkpoints.
- A2C, PPO, and Hierarchical PPO use Stable-Baselines3 `.zip` checkpoints.
- Extra `.pt` policy-state exports may be generated for instructor inspection when supported.
- CSV training logs are written to `training_logs/`.

Example log/artifact tag:

```text
train_100000
```

## Important fixes included

### ARTIFACT_TAG video-export fix

`ARTIFACT_TAG` is defined before the early baseline-video export cell.  
This prevents the error:

```text
NameError: name 'ARTIFACT_TAG' is not defined
```

### Video export flag fix

`EXPORT_COMPARISON_3D_VIDEOS` is defined near the start of the notebook.  
Fallback definitions are also included inside the video diagnostic cell to prevent video diagnostics from failing when cells are run out of order.

## Recommended files to upload to GitHub

Upload the important source files only:

```text
README.md
requirements.txt
*.py
*.ipynb
shooter/
training_logs/
```

Avoid uploading temporary or heavy files:

```text
__pycache__/
.ipynb_checkpoints/
outputs/
videos/
models/
*.mp4
*.pth
*.pt
```

Large videos and model checkpoints should be kept locally unless the instructor specifically asks for them.

## Summary

This 100k package is a quick experimental version of the Shooter-v0 DRL workflow.  
It is useful for debugging, fast evaluation, and generating early evidence, while longer training budgets should be used for stronger final performance.
