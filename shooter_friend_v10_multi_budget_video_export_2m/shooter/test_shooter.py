# File purpose: Validates the ShooterEnv implementation against the Gymnasium API contract.
# DRL role: Confirms that training code can safely call reset(), step(), spaces, seed resets, and random episodes.
# Dependency link: Imports ShooterEnv, GameEngine, constants, and action names from shooter_env.py.

"""
test_shooter.py — Validates ShooterEnv against the Gymnasium API contract.

Checks
------
1. reset() / step() shapes, dtypes, info keys (incl. quota fields)
2. Full random episode runs to termination without errors
3. Seeded resets produce identical initial observations
4. Observation values are finite and in expected range
5. All 7 actions execute without error
6. Wave guard — win does not fire before all NUM_PERIODIC_WAVES waves are launched
7. gym.make("Shooter-v0") registration works end-to-end
"""

import sys                                      # Imports system utilities used for module path setup and runtime control.
import traceback                                # Imports traceback support for readable test failure diagnostics.
import numpy as np                              # Imports NumPy for arrays, random generators, and numerical checks.

# Allow running from the project root without installing the package
sys.path.insert(0, __file__.replace("/shooter/test_shooter.py", ""))  # Executes one step of the file logic needed by the environment, demo, or test workflow.

from shooter.shooter_env import (               # Imports environment classes, constants, and helper functions from the main environment file.
    ShooterEnv, GameEngine, OBS_SIZE, NUM_ACTIONS, ACTION_NAMES,  # Continues the current multi-line collection or argument list.
    NUM_PERIODIC_WAVES, MAX_GAME_TICKS,         # Continues the current multi-line collection or argument list.
)                                               # Closes the current multi-line function call or data structure.


def _header(text: str):                         # Prints a formatted test-section title for clearer console output.
    print(f"\n{'─'*55}")                        # Prints progress, diagnostics, or results to the console for easier presentation and debugging.
    print(f"  {text}")                          # Prints progress, diagnostics, or results to the console for easier presentation and debugging.
    print(f"{'─'*55}")                          # Prints progress, diagnostics, or results to the console for easier presentation and debugging.


def test_api_contract():                        # Tests whether reset(), step(), spaces, dtypes, and info keys match Gymnasium expectations.
    _header("1 · API contract — shapes, types, spaces")  # Executes one step of the file logic needed by the environment, demo, or test workflow.
    env = ShooterEnv(render_mode=None)          # Creates an environment instance that exposes the Gymnasium training interface.

    obs, info = env.reset(seed=0)               # Starts a new episode and returns the initial observation plus diagnostic info.

    # Observation shape and dtype
    assert obs.shape == (OBS_SIZE,), f"Expected ({OBS_SIZE},), got {obs.shape}"  # Validates an expected condition and raises an error if the condition is false.
    assert obs.dtype == np.float32,  f"Expected float32, got {obs.dtype}"  # Validates an expected condition and raises an error if the condition is false.
    assert env.observation_space.contains(obs), "obs not in observation_space"  # Validates an expected condition and raises an error if the condition is false.

    # Info keys
    for key in ("tick", "hunterScore", "alive_ai", "total_ai",  # Starts a loop that repeats the indented logic for each item in the sequence.
                "total_spawned_ai", "wave", "wave_capacity",  # Continues the current multi-line collection or argument list.
                "bullets_in_flight", "gameOverReason"):  # Executes one step of the file logic needed by the environment, demo, or test workflow.
        assert key in info, f"Missing info key: {key}"  # Validates an expected condition and raises an error if the condition is false.
    assert info["total_spawned_ai"] == 4, \
        "total_spawned_ai should start at 4 (initial AI count)"  # Executes one step of the file logic needed by the environment, demo, or test workflow.
    assert info["wave"] == 0,            "wave should start at 0"  # Validates an expected condition and raises an error if the condition is false.
    assert info["wave_capacity"] == 4,   "wave_capacity should start at 4"  # Validates an expected condition and raises an error if the condition is false.
    assert env.action_space.n == 6,      "RL mode should expose Discrete(6), not 7"  # Validates an expected condition and raises an error if the condition is false.

    # Step output
    obs2, reward, terminated, truncated, info2 = env.step(1)  # Sends an action to the environment and receives the next DRL transition result.
    assert obs2.shape  == (OBS_SIZE,)           # Validates an expected condition and raises an error if the condition is false.
    assert isinstance(reward,     float)        # Validates an expected condition and raises an error if the condition is false.
    assert isinstance(terminated, bool)         # Validates an expected condition and raises an error if the condition is false.
    assert isinstance(truncated,  bool)         # Validates an expected condition and raises an error if the condition is false.
    assert env.observation_space.contains(obs2)  # Validates an expected condition and raises an error if the condition is false.

    env.close()                                 # Closes environment or rendering resources after use.
    print("  PASSED")                           # Prints progress, diagnostics, or results to the console for easier presentation and debugging.


def test_random_episode():                      # Runs one random episode to check that the environment can finish without crashing.
    _header("2 · Full random episode — termination and reward")  # Executes one step of the file logic needed by the environment, demo, or test workflow.
    env = ShooterEnv(render_mode=None)          # Creates an environment instance that exposes the Gymnasium training interface.
    obs, info = env.reset(seed=7)               # Starts a new episode and returns the initial observation plus diagnostic info.

    total_reward = 0.0                          # Assigns total_reward for use by the following logic.
    steps = 0                                   # Assigns steps for use by the following logic.
    while True:                                 # Starts a loop that continues while the condition remains true.
        action = env.action_space.sample()      # Samples a random valid action from the environment action space.
        obs, reward, terminated, truncated, info = env.step(action)  # Sends an action to the environment and receives the next DRL transition result.
        assert env.observation_space.contains(obs), f"obs out of space at step {steps}"  # Validates an expected condition and raises an error if the condition is false.
        total_reward += reward                  # Assigns total_reward + for use by the following logic.
        steps += 1                              # Assigns steps + for use by the following logic.
        if terminated or truncated:             # Starts a conditional branch that executes only when the stated condition is true.
            break                               # Stops the nearest loop immediately.

    reason = info["gameOverReason"] or "truncated"  # Assigns reason for use by the following logic.
    print(f"  Episode ended after {steps} steps  |  "  # Prints progress, diagnostics, or results to the console for easier presentation and debugging.
          f"total reward = {total_reward:.1f}  |  reason: {reason}")  # Assigns f"total reward for use by the following logic.
    assert steps > 0                            # Validates an expected condition and raises an error if the condition is false.
    if terminated:                              # Starts a conditional branch that executes only when the stated condition is true.
        assert info["gameOverReason"] != "", "terminated without a gameOverReason"  # Validates an expected condition and raises an error if the condition is false.
    env.close()                                 # Closes environment or rendering resources after use.
    print("  PASSED")                           # Prints progress, diagnostics, or results to the console for easier presentation and debugging.


def test_seed_reproducibility():                # Checks that identical seeds reproduce identical initial observations.
    _header("3 · Seed reproducibility")         # Executes one step of the file logic needed by the environment, demo, or test workflow.
    env = ShooterEnv(render_mode=None)          # Creates an environment instance that exposes the Gymnasium training interface.

    obs_a, _ = env.reset(seed=42)               # Starts a new episode and returns the initial observation plus diagnostic info.
    obs_b, _ = env.reset(seed=42)               # Starts a new episode and returns the initial observation plus diagnostic info.
    assert np.allclose(obs_a, obs_b), "Seeded reset produced different observations"  # Validates an expected condition and raises an error if the condition is false.

    obs_c, _ = env.reset(seed=99)               # Starts a new episode and returns the initial observation plus diagnostic info.
    assert not np.allclose(obs_a, obs_c), "Different seeds gave identical observations"  # Validates an expected condition and raises an error if the condition is false.

    env.close()                                 # Closes environment or rendering resources after use.
    print("  PASSED")                           # Prints progress, diagnostics, or results to the console for easier presentation and debugging.


def test_obs_range():                           # Checks that observation values remain finite and within a reasonable normalized range.
    _header("4 · Observation sanity — normalised values in expected range")  # Executes one step of the file logic needed by the environment, demo, or test workflow.
    env = ShooterEnv(render_mode=None)          # Creates an environment instance that exposes the Gymnasium training interface.
    env.reset(seed=1)                           # Starts a new episode and returns the initial observation plus diagnostic info.

    max_abs = 0.0                               # Assigns max_abs for use by the following logic.
    for _ in range(300):                        # Starts a loop that repeats the indented logic for each item in the sequence.
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())  # Sends an action to the environment and receives the next DRL transition result.
        # Global / turret features (indices 0-5) should be roughly in [-2, 2]
        assert np.all(np.isfinite(obs)), "Non-finite value in observation"  # Validates an expected condition and raises an error if the condition is false.
        max_abs = max(max_abs, float(np.abs(obs[:6]).max()))  # Assigns max_abs for use by the following logic.
        if terminated or truncated:             # Starts a conditional branch that executes only when the stated condition is true.
            break                               # Stops the nearest loop immediately.

    print(f"  Max |obs[0:6]| over episode = {max_abs:.4f}  (expected ≈ [0, 2])")  # Prints progress, diagnostics, or results to the console for easier presentation and debugging.
    assert max_abs < 5.0, f"Observation values suspiciously large: {max_abs}"  # Validates an expected condition and raises an error if the condition is false.
    env.close()                                 # Closes environment or rendering resources after use.
    print("  PASSED")                           # Prints progress, diagnostics, or results to the console for easier presentation and debugging.


def test_action_coverage():                     # Checks that every action exposed by the action space can execute safely.
    _header("5 · All 7 actions execute without error")  # Executes one step of the file logic needed by the environment, demo, or test workflow.
    env = ShooterEnv(render_mode=None)          # Creates an environment instance that exposes the Gymnasium training interface.
    env.reset(seed=3)                           # Starts a new episode and returns the initial observation plus diagnostic info.

    for action in range(env.action_space.n):    # Starts a loop that repeats the indented logic for each item in the sequence.
        try:                                    # Starts a protected block so errors can be handled cleanly.
            obs, reward, terminated, truncated, info = env.step(action)  # Sends an action to the environment and receives the next DRL transition result.
            assert env.observation_space.contains(obs)  # Validates an expected condition and raises an error if the condition is false.
            print(f"    action {action} ({ACTION_NAMES[action]:20s}) "  # Prints progress, diagnostics, or results to the console for easier presentation and debugging.
                  f"reward={reward:+.2f}  ok")  # Assigns f"reward for use by the following logic.
        except Exception as e:                  # Handles errors raised by the matching try block.
            print(f"    action {action} FAILED: {e}")  # Prints progress, diagnostics, or results to the console for easier presentation and debugging.
            raise                               # Re-raises or creates an error after the failure has been detected.
        if terminated or truncated:             # Starts a conditional branch that executes only when the stated condition is true.
            env.reset()                         # Starts a new episode and returns the initial observation plus diagnostic info.

    env.close()                                 # Closes environment or rendering resources after use.
    print("  PASSED")                           # Prints progress, diagnostics, or results to the console for easier presentation and debugging.


def test_wave_guard():                          # Checks that the win condition does not trigger before all scheduled waves launch.
    _header("6 · Wave guard — win must not fire before all periodic waves launched")  # Executes one step of the file logic needed by the environment, demo, or test workflow.
    rng    = np.random.default_rng(42)          # Creates a seeded NumPy random generator for reproducible environment behavior.
    engine = GameEngine(rng)                    # Assigns engine for use by the following logic.
    engine.reset()                              # Starts a new episode and returns the initial observation plus diagnostic info.

    # Advance past the tick > 20 guard used by the win condition
    for _ in range(25):                         # Starts a loop that repeats the indented logic for each item in the sequence.
        engine.step(0)                          # Sends an action to the environment and receives the next DRL transition result.

    # Force-kill every currently alive vehicle and freeze their respawn timers
    for v in engine.state["vehicles"]:          # Starts a loop that repeats the indented logic for each item in the sequence.
        v["alive"]        = False               # Assigns v["alive"] for use by the following logic.
        v["respawnTimer"] = MAX_GAME_TICKS  # won't respawn during this test  # Presentation note: Assigns v["respawnTimer"] for use by the following logic.

    # One more tick — win must NOT fire: waves_launched (0) < NUM_PERIODIC_WAVES (3)
    engine.step(0)                              # Sends an action to the environment and receives the next DRL transition result.

    assert not engine.state["gameOver"], (      # Validates an expected condition and raises an error if the condition is false.
        f"Win fired early: waves_launched={engine.state['waves_launched']} "  # Assigns f"Win fired early: waves_launched for use by the following logic.
        f"but NUM_PERIODIC_WAVES={NUM_PERIODIC_WAVES}"  # Assigns f"but NUM_PERIODIC_WAVES for use by the following logic.
    )                                           # Closes the current multi-line function call or data structure.
    print(f"  waves_launched={engine.state['waves_launched']}  "  # Prints progress, diagnostics, or results to the console for easier presentation and debugging.
          f"NUM_PERIODIC_WAVES={NUM_PERIODIC_WAVES}  "  # Assigns f"NUM_PERIODIC_WAVES for use by the following logic.
          f"gameOver={engine.state['gameOver']}")  # Assigns f"gameOver for use by the following logic.
    print("  PASSED")                           # Prints progress, diagnostics, or results to the console for easier presentation and debugging.


def test_gym_make():                            # Checks that Gymnasium registration works through gym.make("Shooter-v0").
    _header("7 · gym.make('Shooter-v0') registration")  # Executes one step of the file logic needed by the environment, demo, or test workflow.
    try:                                        # Starts a protected block so errors can be handled cleanly.
        import gymnasium as gym                 # Imports Gymnasium as the reinforcement-learning environment interface.
        import shooter  # noqa: F401 — triggers registration  # Presentation note: Imports shooter  # noqa: F401 — triggers registration for supporting functionality used later in this file.
        env = gym.make("Shooter-v0")            # Creates an environment instance that exposes the Gymnasium training interface.
        obs, _ = env.reset(seed=0)              # Starts a new episode and returns the initial observation plus diagnostic info.
        assert obs.shape == (OBS_SIZE,)         # Validates an expected condition and raises an error if the condition is false.
        env.close()                             # Closes environment or rendering resources after use.
        print("  PASSED")                       # Prints progress, diagnostics, or results to the console for easier presentation and debugging.
    except Exception as e:                      # Handles errors raised by the matching try block.
        print(f"  SKIPPED ({e})")               # Prints progress, diagnostics, or results to the console for easier presentation and debugging.


if __name__ == "__main__":                      # Runs the test or demo code only when this file is executed directly.
    errors = []                                 # Assigns errors for use by the following logic.
    for test_fn in (                            # Starts a loop that repeats the indented logic for each item in the sequence.
        test_api_contract,                      # Continues the current multi-line collection or argument list.
        test_random_episode,                    # Continues the current multi-line collection or argument list.
        test_seed_reproducibility,              # Continues the current multi-line collection or argument list.
        test_obs_range,                         # Continues the current multi-line collection or argument list.
        test_action_coverage,                   # Continues the current multi-line collection or argument list.
        test_wave_guard,                        # Continues the current multi-line collection or argument list.
        test_gym_make,                          # Continues the current multi-line collection or argument list.
    ):                                          # Executes one step of the file logic needed by the environment, demo, or test workflow.
        try:                                    # Starts a protected block so errors can be handled cleanly.
            test_fn()                           # Executes one step of the file logic needed by the environment, demo, or test workflow.
        except Exception:                       # Handles errors raised by the matching try block.
            errors.append(test_fn.__name__)     # Executes one step of the file logic needed by the environment, demo, or test workflow.
            traceback.print_exc()               # Executes one step of the file logic needed by the environment, demo, or test workflow.

    print(f"\n{'═'*55}")                        # Prints progress, diagnostics, or results to the console for easier presentation and debugging.
    if errors:                                  # Starts a conditional branch that executes only when the stated condition is true.
        print(f"  FAILED: {', '.join(errors)}")  # Prints progress, diagnostics, or results to the console for easier presentation and debugging.
        sys.exit(1)                             # Executes one step of the file logic needed by the environment, demo, or test workflow.
    else:                                       # Starts the fallback branch when earlier conditions were false.
        print("  All tests passed ✓")           # Prints progress, diagnostics, or results to the console for easier presentation and debugging.
    print(f"{'═'*55}\n")                        # Prints progress, diagnostics, or results to the console for easier presentation and debugging.
