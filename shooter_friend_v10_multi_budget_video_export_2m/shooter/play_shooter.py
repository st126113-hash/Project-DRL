# File purpose: Runs an interactive keyboard demo for the ShooterEnv environment.
# DRL role: Helps inspect environment mechanics manually before training an agent.
# Dependency link: Imports ShooterEnv and rendering helper functions from shooter_env.py.

"""
play_shooter.py — Interactive keyboard play for the 3D ShooterEnv.

Launches the PyOpenGL window, lets you drive the Hunter turret with the
keyboard, and composites a semi-transparent controls + stats panel onto
the rendered frame via env._pre_flip_hook — called once per tick, just
before pygame.display.flip(), so there is no double-flip flicker.

Controls
--------
  ← / → (or A / D)   yaw left / right
  ↑ / ↓ (or W / S)   pitch up / down
  SPACE               fire
  F                   fire at nearest vehicle (auto-aim + fire)
  R                   reset episode
  ESC / Q             quit

Usage
-----
  python shooter/play_shooter.py [--seed SEED]
"""

import sys                                      # Imports system utilities used for module path setup and runtime control.
import argparse                                 # Imports command-line argument parsing for configurable script execution.
import pygame                                   # Imports pygame for keyboard input, window events, and font rendering.

sys.path.insert(0, __file__.replace("/shooter/play_shooter.py", ""))  # Executes one step of the file logic needed by the environment, demo, or test workflow.

from shooter.shooter_env import (               # Imports environment classes, constants, and helper functions from the main environment file.
    ShooterEnv, ACTION_NAMES, NUM_PERIODIC_WAVES,  # Continues the current multi-line collection or argument list.
    gl_begin_2d, gl_end_2d, gl_fill_rect, gl_blit,  # Switches OpenGL into 2D overlay mode for screen-space drawing.
)                                               # Closes the current multi-line function call or data structure.


# ── Key → discrete-action mapping ─────────────────────────────────────────

_KEY_ACTION = [                                 # Defines constant _KEY_ACTION used to control environment behavior consistently.
    (pygame.K_SPACE,  1),                       # Continues a multi-line collection, tuple, or function argument list.
    (pygame.K_f,      6),                       # Continues a multi-line collection, tuple, or function argument list.
    (pygame.K_LEFT,   2), (pygame.K_a, 2),      # Continues a multi-line collection, tuple, or function argument list.
    (pygame.K_RIGHT,  3), (pygame.K_d, 3),      # Continues a multi-line collection, tuple, or function argument list.
    (pygame.K_UP,     4), (pygame.K_w, 4),      # Continues a multi-line collection, tuple, or function argument list.
    (pygame.K_DOWN,   5), (pygame.K_s, 5),      # Continues a multi-line collection, tuple, or function argument list.
]                                               # Executes one step of the file logic needed by the environment, demo, or test workflow.

_KEY_HELP = [                                   # Defines constant _KEY_HELP used to control environment behavior consistently.
    ("← / A",   "yaw left"),                    # Continues a multi-line collection, tuple, or function argument list.
    ("→ / D",   "yaw right"),                   # Continues a multi-line collection, tuple, or function argument list.
    ("↑ / W",   "pitch up"),                    # Continues a multi-line collection, tuple, or function argument list.
    ("↓ / S",   "pitch down"),                  # Continues a multi-line collection, tuple, or function argument list.
    ("SPACE",   "fire"),                        # Continues a multi-line collection, tuple, or function argument list.
    ("F",       "auto-aim + fire"),             # Continues a multi-line collection, tuple, or function argument list.
    ("R",       "reset episode"),               # Continues a multi-line collection, tuple, or function argument list.
    ("ESC / Q", "quit"),                        # Continues a multi-line collection, tuple, or function argument list.
]                                               # Executes one step of the file logic needed by the environment, demo, or test workflow.

_COL_KEY  = 80                                  # Defines constant _COL_KEY used to control environment behavior consistently.
_COL_DESC = 105                                 # Defines constant _COL_DESC used to control environment behavior consistently.
_PANEL_W  = _COL_KEY + _COL_DESC + 16   # 201 px  # Presentation note: Defines constant _PANEL_W used to control environment behavior consistently.
_LINE_H   = 17                                  # Defines constant _LINE_H used to control environment behavior consistently.
_PAD      = 7                                   # Defines constant _PAD used to control environment behavior consistently.


def _read_action(keys) -> int:                  # Maps currently pressed keyboard keys to the discrete action ID used by the environment.
    for key, action in _KEY_ACTION:             # Starts a loop that repeats the indented logic for each item in the sequence.
        if keys[key]:                           # Starts a conditional branch that executes only when the stated condition is true.
            return action                       # Returns the computed value to the caller.
    return 0                                    # Returns the computed value to the caller.


def _draw_overlay(font, hud: dict, W: int) -> None:  # Draws the teaching-style control and statistics overlay on top of the rendered frame.
    """
    Render the controls + live-stats panel using OpenGL 2D helpers.
    Must be called inside env._pre_flip_hook (OpenGL context active,
    gl_begin_2d / gl_end_2d managed here).
    """
    action_name = ACTION_NAMES[hud["action"]]   # Assigns action_name for use by the following logic.
    reward_col  = (100, 255, 130) if hud["reward"] >= 0 else (255, 100, 100)  # Executes one step of the file logic needed by the environment, demo, or test workflow.
    total_col   = (100, 255, 130) if hud["total"]  >= 0 else (255, 100, 100)  # Executes one step of the file logic needed by the environment, demo, or test workflow.

    # Build rows: (key_text, desc_text, key_color, desc_color)
    ctrl_rows = [                               # Assigns ctrl_rows for use by the following logic.
        (k, d, (230, 230, 100), (180, 180, 180)) for k, d in _KEY_HELP  # Continues a multi-line collection, tuple, or function argument list.
    ]                                           # Executes one step of the file logic needed by the environment, demo, or test workflow.
    stat_rows = [                               # Assigns stat_rows for use by the following logic.
        ("Episode",  str(hud["episode"]),              (160, 210, 255), (160, 210, 255)),  # Continues a multi-line collection, tuple, or function argument list.
        ("Action",   action_name,                      (150, 150, 150), (255, 200, 100)),  # Continues a multi-line collection, tuple, or function argument list.
        ("Reward",   f"{hud['reward']:+.2f}",          (150, 150, 150), reward_col),  # Continues a multi-line collection, tuple, or function argument list.
        ("Total R",  f"{hud['total']:+.1f}",           (150, 150, 150), total_col),  # Continues a multi-line collection, tuple, or function argument list.
        ("Score",    str(hud["score"]),                                        (150, 150, 150), (200, 200, 200)),  # Continues a multi-line collection, tuple, or function argument list.
        ("AI alive", str(hud["alive_ai"]),                                     (150, 150, 150), (200, 200, 200)),  # Continues a multi-line collection, tuple, or function argument list.
        ("Wave",     f"{hud['wave']}/{NUM_PERIODIC_WAVES}  cap {hud['wave_capacity']}", (150, 150, 150), (200, 200, 200)),  # Continues a multi-line collection, tuple, or function argument list.
    ]                                           # Executes one step of the file logic needed by the environment, demo, or test workflow.

    # Total rows: header + controls + divider-gap + stats
    n_rows  = 1 + len(ctrl_rows) + 1 + len(stat_rows)  # Assigns n_rows for use by the following logic.
    panel_h = _PAD + n_rows * _LINE_H + _PAD    # Assigns panel_h for use by the following logic.
    px      = W - _PANEL_W - 6                  # Assigns px for use by the following logic.
    py      = W - panel_h - 6   # bottom of panel in Y-up coords  # Presentation note: Assigns py for use by the following logic.

    gl_begin_2d(W, W)                           # Switches OpenGL into 2D overlay mode for screen-space drawing.

    # Background
    gl_fill_rect(px, py, _PANEL_W, panel_h, 0.03, 0.03, 0.03, 0.80)  # Draws a filled rectangle used for UI panels, dividers, or background shapes.

    row = 0                                     # Assigns row for use by the following logic.

    # ── Controls header ───────────────────────────────────────────────────
    y = py + panel_h - _PAD - (row + 1) * _LINE_H  # Assigns y for use by the following logic.
    gl_blit("CONTROLS", px + _PAD, y, font, (255, 215, 60))  # Draws text on the OpenGL surface using a pygame-rendered texture.
    row += 1                                    # Assigns row + for use by the following logic.

    for key_str, desc, kc, dc in ctrl_rows:     # Starts a loop that repeats the indented logic for each item in the sequence.
        y = py + panel_h - _PAD - (row + 1) * _LINE_H  # Assigns y for use by the following logic.
        gl_blit(key_str, px + _PAD,            y, font, kc)  # Draws text on the OpenGL surface using a pygame-rendered texture.
        gl_blit(desc,    px + _PAD + _COL_KEY, y, font, dc)  # Draws text on the OpenGL surface using a pygame-rendered texture.
        row += 1                                # Assigns row + for use by the following logic.

    # ── Divider gap ───────────────────────────────────────────────────────
    div_y = py + panel_h - _PAD - (row + 1) * _LINE_H + _LINE_H // 2  # Assigns div_y for use by the following logic.
    # Use gl_fill_rect as a 1-pixel-high divider line
    gl_fill_rect(px + _PAD, div_y, _PANEL_W - 2 * _PAD, 2,  # Draws a filled rectangle used for UI panels, dividers, or background shapes.
                 0.27, 0.27, 0.27, 1.0)         # Executes one step of the file logic needed by the environment, demo, or test workflow.
    row += 1                                    # Assigns row + for use by the following logic.

    # ── Live stats ────────────────────────────────────────────────────────
    for label, value, lc, vc in stat_rows:      # Starts a loop that repeats the indented logic for each item in the sequence.
        y = py + panel_h - _PAD - (row + 1) * _LINE_H  # Assigns y for use by the following logic.
        gl_blit(label, px + _PAD,            y, font, lc)  # Draws text on the OpenGL surface using a pygame-rendered texture.
        gl_blit(value, px + _PAD + _COL_KEY, y, font, vc)  # Draws text on the OpenGL surface using a pygame-rendered texture.
        row += 1                                # Assigns row + for use by the following logic.

    gl_end_2d()                                 # Restores OpenGL state after drawing the 2D overlay.


def main():                                     # Defines the script entry point that starts the interactive demo loop.
    parser = argparse.ArgumentParser(description="Interactive ShooterEnv play")  # Creates a command-line parser for demo options such as the random seed.
    parser.add_argument("--seed", type=int, default=42)  # Adds a command-line option that changes script behavior without editing the code.
    args = parser.parse_args()                  # Reads command-line arguments and stores them in an easy-to-access object.

    env = ShooterEnv(render_mode="human")       # Creates an environment instance that exposes the Gymnasium training interface.
    obs, info = env.reset(seed=args.seed)       # Starts a new episode and returns the initial observation plus diagnostic info.

    # pygame is initialised inside the first _render_frame call above
    font = pygame.font.SysFont("monospace", 13)  # Creates a font object used to render readable text overlays.
    W    = env._WINDOW_SIZE                     # Defines constant W used to control environment behavior consistently.

    hud = {                                     # Assigns hud for use by the following logic.
        "action":        0,                     # Continues the current multi-line collection or argument list.
        "reward":        0.0,                   # Continues the current multi-line collection or argument list.
        "total":         0.0,                   # Continues the current multi-line collection or argument list.
        "episode":       1,                     # Continues the current multi-line collection or argument list.
        "score":         info["hunterScore"],   # Continues the current multi-line collection or argument list.
        "alive_ai":      info["alive_ai"],      # Continues the current multi-line collection or argument list.
        "wave":          info["wave"],          # Continues the current multi-line collection or argument list.
        "wave_capacity": info["wave_capacity"],  # Continues the current multi-line collection or argument list.
    }                                           # Executes one step of the file logic needed by the environment, demo, or test workflow.

    def _pre_flip() -> None:                    # Defines a hook that draws the overlay immediately before the display buffer is shown.
        _draw_overlay(font, hud, W)             # Executes one step of the file logic needed by the environment, demo, or test workflow.

    env._pre_flip_hook = _pre_flip              # Assigns env._pre_flip_hook for use by the following logic.

    running = True                              # Assigns running for use by the following logic.

    print("═" * 55)                             # Prints progress, diagnostics, or results to the console for easier presentation and debugging.
    print("  Shooter-v0  |  3D Interactive Play")  # Prints progress, diagnostics, or results to the console for easier presentation and debugging.
    print("  ← → ↑ ↓ / WASD   SPACE=fire   F=auto-aim+fire")  # Prints progress, diagnostics, or results to the console for easier presentation and debugging.
    print("  R=reset           ESC/Q=quit")     # Prints progress, diagnostics, or results to the console for easier presentation and debugging.
    print("═" * 55)                             # Prints progress, diagnostics, or results to the console for easier presentation and debugging.

    while running:                              # Starts a loop that continues while the condition remains true.

        # ── 1. Events ─────────────────────────────────────────────────────
        for event in pygame.event.get():        # Starts a loop that repeats the indented logic for each item in the sequence.
            if event.type == pygame.QUIT:       # Starts a conditional branch that executes only when the stated condition is true.
                running = False                 # Assigns running for use by the following logic.
            elif event.type == pygame.KEYDOWN:  # Starts an alternative conditional branch after a previous condition was false.
                if event.key in (pygame.K_ESCAPE, pygame.K_q):  # Starts a conditional branch that executes only when the stated condition is true.
                    running = False             # Assigns running for use by the following logic.
                elif event.key == pygame.K_r:   # Starts an alternative conditional branch after a previous condition was false.
                    obs, info = env.reset()     # Starts a new episode and returns the initial observation plus diagnostic info.
                    hud.update(action=0, reward=0.0, total=0.0, score=0,  # Assigns hud.update(action for use by the following logic.
                               alive_ai=info["alive_ai"],  # Assigns alive_ai for use by the following logic.
                               wave=info["wave"],  # Assigns wave for use by the following logic.
                               wave_capacity=info["wave_capacity"])  # Assigns wave_capacity for use by the following logic.
                    hud["episode"] += 1         # Assigns hud["episode"] + for use by the following logic.
                    print(f"\n[Episode {hud['episode']}]  Manual reset")  # Prints progress, diagnostics, or results to the console for easier presentation and debugging.

        if not running:                         # Starts a conditional branch that executes only when the stated condition is true.
            break                               # Stops the nearest loop immediately.

        # ── 2. Action ─────────────────────────────────────────────────────
        action       = _read_action(pygame.key.get_pressed())  # Reads the current keyboard state for continuous control actions.
        hud["action"] = action                  # Assigns hud["action"] for use by the following logic.

        # ── 3. Step ───────────────────────────────────────────────────────
        obs, reward, terminated, truncated, info = env.step(action)  # Sends an action to the environment and receives the next DRL transition result.

        # ── 4. Update HUD ─────────────────────────────────────────────────
        hud["reward"]        = reward           # Assigns hud["reward"] for use by the following logic.
        hud["total"]        += reward           # Assigns hud["total"]        + for use by the following logic.
        hud["score"]         = info["hunterScore"]  # Assigns hud["score"] for use by the following logic.
        hud["alive_ai"]      = info["alive_ai"]  # Assigns hud["alive_ai"] for use by the following logic.
        hud["wave"]          = info["wave"]     # Assigns hud["wave"] for use by the following logic.
        hud["wave_capacity"] = info["wave_capacity"]  # Assigns hud["wave_capacity"] for use by the following logic.

        # ── 5. Episode end ────────────────────────────────────────────────
        if terminated or truncated:             # Starts a conditional branch that executes only when the stated condition is true.
            reason = info["gameOverReason"] or "truncated"  # Assigns reason for use by the following logic.
            print(                              # Prints progress, diagnostics, or results to the console for easier presentation and debugging.
                f"[Episode {hud['episode']}]  {reason}\n"  # Executes one step of the file logic needed by the environment, demo, or test workflow.
                f"  ticks={info['tick']:5d}  score={info['hunterScore']:4d}"  # Assigns f"  ticks for use by the following logic.
                f"  total_reward={hud['total']:+.1f}"  # Assigns f"  total_reward for use by the following logic.
            )                                   # Closes the current multi-line function call or data structure.
            obs, info = env.reset()             # Starts a new episode and returns the initial observation plus diagnostic info.
            hud.update(action=0, reward=0.0, total=0.0, score=0,  # Assigns hud.update(action for use by the following logic.
                       alive_ai=info["alive_ai"],  # Assigns alive_ai for use by the following logic.
                       wave=info["wave"],       # Assigns wave for use by the following logic.
                       wave_capacity=info["wave_capacity"])  # Assigns wave_capacity for use by the following logic.
            hud["episode"] += 1                 # Assigns hud["episode"] + for use by the following logic.

    env.close()                                 # Closes environment or rendering resources after use.
    print("\nSession ended.")                   # Prints progress, diagnostics, or results to the console for easier presentation and debugging.


if __name__ == "__main__":                      # Runs the test or demo code only when this file is executed directly.
    main()                                      # Executes one step of the file logic needed by the environment, demo, or test workflow.
