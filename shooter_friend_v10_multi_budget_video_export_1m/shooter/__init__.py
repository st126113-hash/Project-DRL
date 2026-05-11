# File purpose: Registers the custom Gymnasium environment name Shooter-v0.
# DRL role: Connects gym.make("Shooter-v0") to the ShooterEnv class in shooter_env.py.
# Dependency link: This file must be imported before gym.make can discover the environment.

from gymnasium.envs.registration import register  # Imports Gymnasium registration so the custom environment can receive a public environment ID.

register(                                       # Starts the Gymnasium registration call for the custom Shooter environment.
    id="Shooter-v0",                            # Defines the environment ID used later by gym.make().
    entry_point="shooter.shooter_env:ShooterEnv",  # Points Gymnasium to the Python class that creates the environment instance.
)                                               # Closes the current multi-line function call or data structure.
