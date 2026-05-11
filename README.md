# Project-DRL: Shooter Hierarchical PPO Experiments

This repository contains Deep Reinforcement Learning (DRL) experiments for the `Shooter-v0` environment.  
The project compares several training budgets and stores notebooks, checkpoints, logs, outputs, and exported videos for each run.

## Project overview

The final approach is **Hierarchical PPO with a deterministic Lead-Aim Controller**.  
The controller supports aiming and shooting decisions, while PPO is used for learning the high-level behavior.  
For fair evaluation, the final low-level action sent to the environment uses actions **0–5** only.

## Uploaded experiment folders

| Folder | Training budget | Purpose |
|---|---:|---|
| `shooter_friend_v10_multi_budget_video_export_100k/` | 100,000 timesteps | Very fast test run and upload proof |
| `shooter_friend_v10_multi_budget_video_export_250k/` | 250,000 timesteps | Fast backup run and practical tuning evidence |
| `shooter_friend_v10_multi_budget_video_export_500k/` | 500,000 timesteps | Medium run for stronger comparison |
| `shooter_friend_v10_multi_budget_video_export_1m/` | 1,000,000 timesteps | Longer final-agent check |
| `shooter_friend_v10_multi_budget_video_export_2m/` | 2,000,000 timesteps | Long training run for stronger final evidence |

## Common folder structure

Each experiment folder follows a similar structure:

```text
checkpoints/               Saved model checkpoints
notebooks/                 Training and evaluation notebooks
outputs/                   Generated outputs and videos
reference/                 Reference files or supporting materials
sample_uploaded_results/   Sample exported results
shooter/                   Shooter-v0 environment code
training_logs/             CSV logs and evaluation summaries
README_*.md                Budget-specific README
RUN_ORDER_AND_ARTIFACTS.md Run order and artifact notes
requirements_notebook.txt  Notebook dependencies
```

## Main methods

The repository includes or references the following DRL methods:

- Dueling Double DQN
- A2C
- PPO
- Hierarchical PPO
- Deterministic Lead-Aim Controller baseline

## Artifact formats

- Dueling Double DQN checkpoints use PyTorch `.pt` format.
- A2C, PPO, and Hierarchical PPO checkpoints use Stable-Baselines3 `.zip` format.
- Training logs are stored as `.csv` files inside `training_logs/`.
- Exported videos are stored inside `outputs/` when video export is enabled.

## Video export

Some notebooks include video export for:

- Lead-Aim baseline
- Final Hierarchical PPO agent
- Comparison videos across methods

Video export requires the local render path to work correctly with **Pygame**, **PyOpenGL**, and the Shooter-v0 environment.

## How to run

Install dependencies:

```bash
pip install -r requirements_notebook.txt
```

Then open the notebook inside the selected budget folder:

```text
notebooks/
```

Run the cells in order, following the instructions in:

```text
RUN_ORDER_AND_ARTIFACTS.md
```

## Notes

This repository is organized mainly for experiment evidence and course submission.  
The separate budget folders allow the same workflow to be compared under different training lengths.
