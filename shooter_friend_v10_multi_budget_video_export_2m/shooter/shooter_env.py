# File purpose: Defines the 3D Shooter game engine and wraps it as a Gymnasium environment.
# DRL role: Provides reset(), step(action), observation_space, action_space, rewards, termination rules, and rendering.
# Dependency link: Imported by __init__.py for registration, by play_shooter.py for manual play, and by test_shooter.py for validation.

"""
shooter_env.py
==============
Hunter-only 3D game engine wrapped as a Farama Gymnasium environment.
Rendered with PyOpenGL — a stationary turret at the origin shoots down
AI drones flying at various altitudes across a 200 × 200 arena.

┌────────────────────────────────────────────────────────────────┐
│  INPUTS  (what the DRL agent sends to the environment)        │
│                                                                │
│  action : int ∈ {0 … 5}   →   Discrete(6) action space       │
│                            (render_mode="human" adds action 6)│
│                                                                │
│   0  do_nothing          hold current aim                     │
│   1  fire                shoot along current aim vector       │
│   2  yaw_left            rotate turret left   (+0.10 rad)     │
│   3  yaw_right           rotate turret right  (−0.10 rad)     │
│   4  pitch_up            tilt barrel up       (+0.02 rad)     │
│   5  pitch_down          tilt barrel down     (−0.02 rad)     │
│   6  fire_at_nearest     auto-aim + fire  [human mode only]   │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│  OUTPUTS  (what the environment returns to the agent)         │
│                                                                │
│  obs        float32[169]  normalised observation vector       │
│  reward     float         shaped scalar reward signal         │
│  terminated bool          game-over event occurred            │
│  truncated  bool          MAX_GAME_TICKS reached              │
│  info       dict          raw diagnostics for logging         │
└────────────────────────────────────────────────────────────────┘

Observation layout  (169 features, all float32)
───────────────────────────────────────────────
  [0]        tick / 4 800
  [1]        hunterScore / 500
  [2]        alive vehicle count / 15  (MAX_VEH_SLOTS; max alive ≈ 13 at wave 3)
  [3]        hunterYaw   / π
  [4]        hunterPitch / 0.9
  [5]        hunterRoll  / 0.25
  [6 …125]   15 vehicle slots × 8 features (sorted nearest-first, zero-padded)
               x/100, z/100, dist/141, sin(angle_xz), cos(angle_xz), y/20, alive, isAI
  [126…165]  10 bullet  slots × 4 features (zero-padded)
               x/100, z/100, dx, dz
  [166…168]  nearest-threat summary
               3d_dist/141, xz_angle/π, approach-speed proxy

Reward components
─────────────────
  +20   × kills this tick
  −2    per shot (actions 1 and 6)
  −(20 − dist) × 0.05   when nearest vehicle is within 20 units (XZ)
  −100  game-over: a vehicle reached the Hunter
  +200  game-over: all waves cleared AND all AI eliminated before time limit
        (no bonus for simply surviving to the tick limit)

Quick start
───────────
    import gymnasium as gym
    import shooter                    # registers Shooter-v0
    env = gym.make("Shooter-v0")
    obs, info = env.reset(seed=42)
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()
"""

import math                                     # Imports math functions for angles, distances, and trigonometry.
from typing import Optional                     # Imports type hints used to document optional values and function signatures.

import numpy as np                              # Imports NumPy for arrays, random generators, and numerical checks.
import gymnasium as gym                         # Imports Gymnasium as the reinforcement-learning environment interface.
from gymnasium import spaces                    # Imports selected objects from another module so this file can reuse existing functionality.

try:                                            # Starts a protected block so errors can be handled cleanly.
    from OpenGL.GL import *                     # Imports OpenGL drawing functions used by the 3D renderer.
    from OpenGL.GLU import *                    # Imports OpenGL drawing functions used by the 3D renderer.
    _OPENGL_OK = True                           # Defines constant _OPENGL_OK used to control environment behavior consistently.
except ImportError:                             # Handles errors raised by the matching try block.
    _OPENGL_OK = False                          # Defines constant _OPENGL_OK used to control environment behavior consistently.


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — GAME CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

TICK_RATE          = 20                         # Defines constant TICK_RATE used to control environment behavior consistently.
ARENA              = 200                        # Defines constant ARENA used to control environment behavior consistently.
HALF               = ARENA // 2                 # Defines constant HALF used to control environment behavior consistently.
TREE_R             = 2.5                        # Defines constant TREE_R used to control environment behavior consistently.
VEH_R              = 2.0                        # Defines constant VEH_R used to control environment behavior consistently.
BULLET_SPEED       = 2.5                        # Defines constant BULLET_SPEED used to control environment behavior consistently.
AI_SPEED           = 0.35                       # Defines constant AI_SPEED used to control environment behavior consistently.
RESPAWN_TICKS_AI   = 5 * TICK_RATE              # Defines constant RESPAWN_TICKS_AI used to control environment behavior consistently.
HIT_DIST           = 3.5                        # Defines constant HIT_DIST used to control environment behavior consistently.
REACH_DIST         = 6.5                        # Defines constant REACH_DIST used to control environment behavior consistently.
BULLET_MAX_DIST    = 250                        # Defines constant BULLET_MAX_DIST used to control environment behavior consistently.
MAX_GAME_TICKS     = 4 * 60 * TICK_RATE    # 4800 ticks = 4 minutes  # Presentation note: Defines constant MAX_GAME_TICKS used to control environment behavior consistently.
WAVE_INTERVAL      = 60 * TICK_RATE        # new wave every 60 s  (ticks: 1200, 2400, 3600)  # Presentation note: Defines constant WAVE_INTERVAL used to control environment behavior consistently.
WAVE_GROWTH        = 1.5                   # each wave is 50% larger than the previous  # Presentation note: Defines constant WAVE_GROWTH used to control environment behavior consistently.
NUM_PERIODIC_WAVES = (MAX_GAME_TICKS // WAVE_INTERVAL) - 1   # = 3  (minutes 1, 2, 3)  # Presentation note: Defines constant NUM_PERIODIC_WAVES used to control environment behavior consistently.
INITIAL_WAVE_SIZE  = 4                     # vehicles at episode start  (= len(_AI_COLORS))  # Presentation note: Defines constant INITIAL_WAVE_SIZE used to control environment behavior consistently.
TREE_CLEAR_RADIUS  = 20                    # no trees within this XZ radius of the turret  # Presentation note: Defines constant TREE_CLEAR_RADIUS used to control environment behavior consistently.
VEHICLE_HEIGHT_MIN = 4.0                   # drone altitude range (units)  # Presentation note: Defines constant VEHICLE_HEIGHT_MIN used to control environment behavior consistently.
VEHICLE_HEIGHT_MAX = 10.0                       # Defines constant VEHICLE_HEIGHT_MAX used to control environment behavior consistently.

_PITCH_MIN, _PITCH_MAX = -0.3, 0.9        # extended range for 3D aiming  # Presentation note: Defines constant _PITCH_MIN, _PITCH_MAX used to control environment behavior consistently.


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — ACTION SPACE
# ═══════════════════════════════════════════════════════════════════════════

NUM_ACTIONS  = 7                                # Defines constant NUM_ACTIONS used to control environment behavior consistently.
ACTION_NAMES = {                                # Defines constant ACTION_NAMES used to control environment behavior consistently.
    0: "do_nothing",                            # Continues the current multi-line collection or argument list.
    1: "fire",                                  # Continues the current multi-line collection or argument list.
    2: "yaw_left",          # Δyaw   = +0.10 rad  # Presentation note: Assigns 2: "yaw_left",          # Δyaw for use by the following logic.
    3: "yaw_right",         # Δyaw   = −0.10 rad  # Presentation note: Assigns 3: "yaw_right",         # Δyaw for use by the following logic.
    4: "pitch_up",          # Δpitch = +0.02 rad  # Presentation note: Assigns 4: "pitch_up",          # Δpitch for use by the following logic.
    5: "pitch_down",        # Δpitch = −0.02 rad  # Presentation note: Assigns 5: "pitch_down",        # Δpitch for use by the following logic.
    6: "fire_at_nearest",   # auto-aim (yaw + pitch) + fire  [human mode only]  # Presentation note: Executes one step of the file logic needed by the environment, demo, or test workflow.
}                                               # Executes one step of the file logic needed by the environment, demo, or test workflow.


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — OBSERVATION SPACE  (169 float32 values)
# ═══════════════════════════════════════════════════════════════════════════

MAX_VEH_SLOTS    = 15                           # Defines constant MAX_VEH_SLOTS used to control environment behavior consistently.
MAX_BULLET_SLOTS = 10                           # Defines constant MAX_BULLET_SLOTS used to control environment behavior consistently.
OBS_SIZE = (                                    # Defines constant OBS_SIZE used to control environment behavior consistently.
    3                        # global state  # Presentation note: Executes one step of the file logic needed by the environment, demo, or test workflow.
    + 3                      # turret state  # Presentation note: Executes one step of the file logic needed by the environment, demo, or test workflow.
    + MAX_VEH_SLOTS    * 8   # 15 × 8 = 120  (added y feature)  # Presentation note: Defines constant + MAX_VEH_SLOTS    * 8   # 15 × 8 used to control environment behavior consistently.
    + MAX_BULLET_SLOTS * 4   # 10 × 4 =  40  # Presentation note: Defines constant + MAX_BULLET_SLOTS * 4   # 10 × 4 used to control environment behavior consistently.
    + 3                      # nearest-threat summary  # Presentation note: Executes one step of the file logic needed by the environment, demo, or test workflow.
)  # = 169  # Presentation note: Assigns )  # for use by the following logic.


class OBS:                                      # Defines named observation-vector indices to avoid using unexplained magic numbers.
    """Named start indices for each block in the observation vector."""
    TICK         = 0                            # Defines constant TICK used to control environment behavior consistently.
    SCORE        = 1                            # Defines constant SCORE used to control environment behavior consistently.
    ALIVE_COUNT  = 2                            # Defines constant ALIVE_COUNT used to control environment behavior consistently.
    YAW          = 3                            # Defines constant YAW used to control environment behavior consistently.
    PITCH        = 4                            # Defines constant PITCH used to control environment behavior consistently.
    ROLL         = 5                            # Defines constant ROLL used to control environment behavior consistently.
    VEH_START    = 6                            # Defines constant VEH_START used to control environment behavior consistently.
    BULLET_START = 6 + MAX_VEH_SLOTS * 8                          # 126  # Presentation note: Defines constant BULLET_START used to control environment behavior consistently.
    THREAT_START = 6 + MAX_VEH_SLOTS * 8 + MAX_BULLET_SLOTS * 4   # 166  # Presentation note: Defines constant THREAT_START used to control environment behavior consistently.


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — PURE GAME-ENGINE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

_AI_COLORS    = ["red", "lime", "cyan", "orange"]  # Defines constant _AI_COLORS used to control environment behavior consistently.
_EXTRA_COLORS = ["magenta", "yellow", "deepskyblue", "tomato", "chartreuse",  # Defines constant _EXTRA_COLORS used to control environment behavior consistently.
                 "gold", "hotpink", "turquoise", "coral", "violet"]  # Executes one step of the file logic needed by the environment, demo, or test workflow.


def _dist(ax: float, az: float, bx: float, bz: float) -> float:  # Defines the _dist function used as a reusable operation in this file.
    return math.sqrt((ax - bx) ** 2 + (az - bz) ** 2)  # Returns the computed value to the caller.


def _dist3(ax: float, ay: float, az: float,     # Defines the _dist3 function used as a reusable operation in this file.
           bx: float, by: float, bz: float) -> float:  # Executes one step of the file logic needed by the environment, demo, or test workflow.
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)  # Returns the computed value to the caller.


def _wrap_yaw(y: float) -> float:               # Defines the _wrap_yaw function used as a reusable operation in this file.
    return y - 2 * math.pi * math.floor((y + math.pi) / (2 * math.pi))  # Returns the computed value to the caller.


def _gen_trees(rng: np.random.Generator) -> list:  # Defines the _gen_trees function used as a reusable operation in this file.
    """genTrees() — guarantees exactly 36 trees placed via seeded RNG."""
    t = []                                      # Assigns t for use by the following logic.

    def _rp():                                  # Defines the _rp function used as a reusable operation in this file.
        return float(rng.uniform(-(HALF - 15), HALF - 15))  # Returns the computed value to the caller.

    for _ in range(6):                          # Starts a loop that repeats the indented logic for each item in the sequence.
        cx, cz = _rp(), _rp()                   # Assigns cx, cz for use by the following logic.
        for _ in range(4):                      # Starts a loop that repeats the indented logic for each item in the sequence.
            x = cx + float(rng.uniform(-6, 6))  # Assigns x for use by the following logic.
            z = cz + float(rng.uniform(-6, 6))  # Assigns z for use by the following logic.
            if math.sqrt(x * x + z * z) > TREE_CLEAR_RADIUS:  # Starts a conditional branch that executes only when the stated condition is true.
                t.append({"x": x, "z": z, "r": TREE_R})  # Executes one step of the file logic needed by the environment, demo, or test workflow.

    for _ in range(12):                         # Starts a loop that repeats the indented logic for each item in the sequence.
        x, z = _rp(), _rp()                     # Assigns x, z for use by the following logic.
        if math.sqrt(x * x + z * z) > TREE_CLEAR_RADIUS:  # Starts a conditional branch that executes only when the stated condition is true.
            t.append({"x": x, "z": z, "r": TREE_R})  # Executes one step of the file logic needed by the environment, demo, or test workflow.

    while len(t) < 36:                          # Starts a loop that continues while the condition remains true.
        x, z = _rp(), _rp()                     # Assigns x, z for use by the following logic.
        if math.sqrt(x * x + z * z) > TREE_CLEAR_RADIUS:  # Starts a conditional branch that executes only when the stated condition is true.
            t.append({"x": x, "z": z, "r": TREE_R})  # Executes one step of the file logic needed by the environment, demo, or test workflow.

    return t[:36]                               # Returns the computed value to the caller.


def _tree_blocked_fast(tree_xs: np.ndarray, tree_zs: np.ndarray,  # Defines the _tree_blocked_fast function used as a reusable operation in this file.
                       x: float, z: float, r: float) -> bool:  # Executes one step of the file logic needed by the environment, demo, or test workflow.
    """Vectorised tree collision check (squared-distance, no sqrt)."""
    threshold_sq = (r + TREE_R) ** 2            # Assigns threshold_sq for use by the following logic.
    dx = tree_xs - x                            # Assigns dx for use by the following logic.
    dz = tree_zs - z                            # Assigns dz for use by the following logic.
    return bool(np.any(dx * dx + dz * dz < threshold_sq))  # Returns the computed value to the caller.


def _vehicle_blocked(vehicles: list, x: float, z: float,  # Defines the _vehicle_blocked function used as a reusable operation in this file.
                     r: float, self_id: str) -> bool:  # Executes one step of the file logic needed by the environment, demo, or test workflow.
    for v in vehicles:                          # Starts a loop that repeats the indented logic for each item in the sequence.
        if not v["alive"] or v["id"] == self_id:  # Starts a conditional branch that executes only when the stated condition is true.
            continue                            # Skips the remaining loop body and moves to the next iteration.
        if _dist(x, z, v["x"], v["z"]) < r + VEH_R:  # Starts a conditional branch that executes only when the stated condition is true.
            return True                         # Returns the computed value to the caller.
    return False                                # Returns the computed value to the caller.


def _clamp_arena(x: float, z: float, r: float):  # Defines the _clamp_arena function used as a reusable operation in this file.
    lim = HALF - r                              # Assigns lim for use by the following logic.
    return max(-lim, min(lim, x)), max(-lim, min(lim, z))  # Returns the computed value to the caller.


def _check_stuck(v: dict, tree_xs: np.ndarray, tree_zs: np.ndarray,  # Defines the _check_stuck function used as a reusable operation in this file.
                 rng: np.random.Generator) -> None:  # Executes one step of the file logic needed by the environment, demo, or test workflow.
    if v["stuckCounter"] % 40 == 0 and v["stuckCounter"] > 0:  # Starts a conditional branch that executes only when the stated condition is true.
        if _dist(v["x"], v["z"], v["lastX"], v["lastZ"]) < 1.0:  # Starts a conditional branch that executes only when the stated condition is true.
            att = 0                             # Assigns att for use by the following logic.
            while att < 20:                     # Starts a loop that continues while the condition remains true.
                a  = float(rng.uniform(0, 2 * math.pi))  # Assigns a for use by the following logic.
                tx = math.cos(a) * (HALF - 15)  # Uses trigonometry to convert angles into direction or normalized feature values.
                tz = math.sin(a) * (HALF - 15)  # Uses trigonometry to convert angles into direction or normalized feature values.
                if not _tree_blocked_fast(tree_xs, tree_zs, tx, tz, VEH_R):  # Starts a conditional branch that executes only when the stated condition is true.
                    break                       # Stops the nearest loop immediately.
                att += 1                        # Assigns att + for use by the following logic.
            v["x"], v["z"] = tx, tz             # Assigns v["x"], v["z"] for use by the following logic.
            v["angle"]      = math.atan2(-v["z"], -v["x"])  # Computes an angle from vector components, useful for yaw, pitch, and direction logic.
            v["stuckCounter"] = 0               # Assigns v["stuckCounter"] for use by the following logic.
            return                              # Exits the current function without returning a value.
    if v["stuckCounter"] % 40 == 0:             # Starts a conditional branch that executes only when the stated condition is true.
        v["lastX"] = v["x"]                     # Assigns v["lastX"] for use by the following logic.
        v["lastZ"] = v["z"]                     # Assigns v["lastZ"] for use by the following logic.
    v["stuckCounter"] += 1                      # Assigns v["stuckCounter"] + for use by the following logic.


def _try_move(v: dict, nx: float, nz: float, ignore_trees: bool,  # Defines the _try_move function used as a reusable operation in this file.
              tree_xs: np.ndarray, tree_zs: np.ndarray, vehicles: list) -> bool:  # Executes one step of the file logic needed by the environment, demo, or test workflow.
    blocked = (                                 # Assigns blocked for use by the following logic.
        (not ignore_trees and _tree_blocked_fast(tree_xs, tree_zs, nx, nz, VEH_R))  # Continues a multi-line collection, tuple, or function argument list.
        or _vehicle_blocked(vehicles, nx, nz, VEH_R, v["id"])  # Executes one step of the file logic needed by the environment, demo, or test workflow.
    )                                           # Closes the current multi-line function call or data structure.
    if not blocked:                             # Starts a conditional branch that executes only when the stated condition is true.
        v["x"], v["z"] = _clamp_arena(nx, nz, VEH_R)  # Assigns v["x"], v["z"] for use by the following logic.
        return True                             # Returns the computed value to the caller.
    return False                                # Returns the computed value to the caller.


def _steer_ai(v: dict, tree_xs: np.ndarray, tree_zs: np.ndarray,  # Defines the _steer_ai function used as a reusable operation in this file.
              vehicles: list, rng: np.random.Generator) -> None:  # Executes one step of the file logic needed by the environment, demo, or test workflow.
    if not v["alive"]:                          # Starts a conditional branch that executes only when the stated condition is true.
        return                                  # Exits the current function without returning a value.
    d = _dist(v["x"], v["z"], 0.0, 0.0)         # Assigns d for use by the following logic.
    _check_stuck(v, tree_xs, tree_zs, rng)      # Executes one step of the file logic needed by the environment, demo, or test workflow.

    desired    = math.atan2(-v["z"], -v["x"])   # Computes an angle from vector components, useful for yaw, pitch, and direction logic.
    random_amt = 0.05 if d < 20 else 0.25       # Assigns random_amt for use by the following logic.
    desired   += float(rng.uniform(-random_amt / 2, random_amt / 2))  # Assigns desired   + for use by the following logic.

    if d > 15:                                  # Starts a conditional branch that executes only when the stated condition is true.
        look_dist = 8                           # Assigns look_dist for use by the following logic.
        ax = v["x"] + math.cos(desired) * look_dist  # Uses trigonometry to convert angles into direction or normalized feature values.
        az = v["z"] + math.sin(desired) * look_dist  # Uses trigonometry to convert angles into direction or normalized feature values.
        if _tree_blocked_fast(tree_xs, tree_zs, ax, az, VEH_R):  # Starts a conditional branch that executes only when the stated condition is true.
            found, offset = False, 0.4          # Assigns found, offset for use by the following logic.
            while offset < math.pi:             # Starts a loop that continues while the condition remains true.
                for sign in (1, -1):            # Starts a loop that repeats the indented logic for each item in the sequence.
                    tx = v["x"] + math.cos(desired + offset * sign) * look_dist  # Uses trigonometry to convert angles into direction or normalized feature values.
                    tz = v["z"] + math.sin(desired + offset * sign) * look_dist  # Uses trigonometry to convert angles into direction or normalized feature values.
                    if not _tree_blocked_fast(tree_xs, tree_zs, tx, tz, VEH_R):  # Starts a conditional branch that executes only when the stated condition is true.
                        desired += offset * sign  # Assigns desired + for use by the following logic.
                        found = True            # Assigns found for use by the following logic.
                        break                   # Stops the nearest loop immediately.
                if found:                       # Starts a conditional branch that executes only when the stated condition is true.
                    break                       # Stops the nearest loop immediately.
                offset += 0.25                  # Assigns offset + for use by the following logic.

    diff = desired - v["angle"]                 # Assigns diff for use by the following logic.
    while diff >  math.pi: diff -= 2 * math.pi  # Starts a loop that continues while the condition remains true.
    while diff < -math.pi: diff += 2 * math.pi  # Starts a loop that continues while the condition remains true.
    turn_rate   = 0.18 if d < 15 else 0.08      # Assigns turn_rate for use by the following logic.
    v["angle"] += math.copysign(min(abs(diff), turn_rate), diff)  # Assigns v["angle"] + for use by the following logic.

    speed = AI_SPEED * 1.4 if d < 20 else AI_SPEED  # Assigns speed for use by the following logic.
    nx    = v["x"] + math.cos(v["angle"]) * speed  # Uses trigonometry to convert angles into direction or normalized feature values.
    nz    = v["z"] + math.sin(v["angle"]) * speed  # Uses trigonometry to convert angles into direction or normalized feature values.

    ignore = d < 15                             # Assigns ignore for use by the following logic.
    if not _try_move(v, nx, nz, ignore, tree_xs, tree_zs, vehicles):  # Starts a conditional branch that executes only when the stated condition is true.
        for off in (0.3, -0.3, 0.6, -0.6, 1.0, -1.0):  # Starts a loop that repeats the indented logic for each item in the sequence.
            if _try_move(v,                     # Starts a conditional branch that executes only when the stated condition is true.
                         v["x"] + math.cos(v["angle"] + off) * speed,  # Uses trigonometry to convert angles into direction or normalized feature values.
                         v["z"] + math.sin(v["angle"] + off) * speed,  # Uses trigonometry to convert angles into direction or normalized feature values.
                         ignore, tree_xs, tree_zs, vehicles):  # Executes one step of the file logic needed by the environment, demo, or test workflow.
                break                           # Stops the nearest loop immediately.


# ═══════════════════════════════════════════════════════════════════════════
# SECTION GL — OPENGL DRAWING UTILITIES
# Module-level; importable by play_shooter.py for overlay rendering.
# All functions must be called with a valid OpenGL context active.
# ═══════════════════════════════════════════════════════════════════════════

def gl_begin_2d(W: int, H: int) -> None:        # Defines the gl_begin_2d function used as a reusable operation in this file.
    """Enter 2D screen-space rendering.  Y=0 at bottom, Y=H at top."""
    glMatrixMode(GL_PROJECTION)                 # Executes an OpenGL rendering operation for the visual environment.
    glPushMatrix()                              # Executes an OpenGL rendering operation for the visual environment.
    glLoadIdentity()                            # Executes an OpenGL rendering operation for the visual environment.
    glOrtho(0, W, 0, H, -1, 1)                  # Executes an OpenGL rendering operation for the visual environment.
    glMatrixMode(GL_MODELVIEW)                  # Executes an OpenGL rendering operation for the visual environment.
    glPushMatrix()                              # Executes an OpenGL rendering operation for the visual environment.
    glLoadIdentity()                            # Executes an OpenGL rendering operation for the visual environment.
    glDisable(GL_DEPTH_TEST)                    # Executes an OpenGL rendering operation for the visual environment.


def gl_end_2d() -> None:                        # Defines the gl_end_2d function used as a reusable operation in this file.
    """Exit 2D screen-space rendering and restore 3D matrices."""
    glEnable(GL_DEPTH_TEST)                     # Executes an OpenGL rendering operation for the visual environment.
    glMatrixMode(GL_PROJECTION)                 # Executes an OpenGL rendering operation for the visual environment.
    glPopMatrix()                               # Executes an OpenGL rendering operation for the visual environment.
    glMatrixMode(GL_MODELVIEW)                  # Executes an OpenGL rendering operation for the visual environment.
    glPopMatrix()                               # Executes an OpenGL rendering operation for the visual environment.


def gl_fill_rect(x: int, y: int, w: int, h: int,  # Defines the gl_fill_rect function used as a reusable operation in this file.
                 r: float, g: float, b: float, a: float = 0.75) -> None:  # Assigns r: float, g: float, b: float, a: float for use by the following logic.
    """Draw a filled axis-aligned rectangle in the current 2D mode."""
    glColor4f(r, g, b, a)                       # Executes an OpenGL rendering operation for the visual environment.
    glBegin(GL_QUADS)                           # Executes an OpenGL rendering operation for the visual environment.
    glVertex2f(x,     y    )                    # Executes an OpenGL rendering operation for the visual environment.
    glVertex2f(x + w, y    )                    # Executes an OpenGL rendering operation for the visual environment.
    glVertex2f(x + w, y + h)                    # Executes an OpenGL rendering operation for the visual environment.
    glVertex2f(x,     y + h)                    # Executes an OpenGL rendering operation for the visual environment.
    glEnd()                                     # Executes an OpenGL rendering operation for the visual environment.


def gl_blit(text: str, x: int, y: int, font, color=(210, 210, 210)) -> int:  # Defines the gl_blit function used as a reusable operation in this file.
    """
    Render a text string at screen position (x, y) using a pygame font.
    y is the BOTTOM edge of the text (Y=0 at screen bottom).
    Returns the rendered text height in pixels.
    Requires gl_begin_2d() to be active.
    """
    import pygame                               # Imports pygame for keyboard input, window events, and font rendering.
    surf = font.render(text, True, color)       # Assigns surf for use by the following logic.
    tw, th = surf.get_size()                    # Assigns tw, th for use by the following logic.
    # True = flip vertically so row-0 is at the bottom (OpenGL convention)
    raw = pygame.image.tostring(surf, "RGBA", True)  # Assigns raw for use by the following logic.

    tex = glGenTextures(1)                      # Assigns tex for use by the following logic.
    glBindTexture(GL_TEXTURE_2D, tex)           # Executes an OpenGL rendering operation for the visual environment.
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)  # Executes an OpenGL rendering operation for the visual environment.
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)  # Executes an OpenGL rendering operation for the visual environment.
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tw, th, 0,  # Continues the current multi-line collection or argument list.
                 GL_RGBA, GL_UNSIGNED_BYTE, raw)  # Executes one step of the file logic needed by the environment, demo, or test workflow.

    glEnable(GL_TEXTURE_2D)                     # Executes an OpenGL rendering operation for the visual environment.
    glColor4f(1.0, 1.0, 1.0, 1.0)               # Executes an OpenGL rendering operation for the visual environment.
    glBegin(GL_QUADS)                           # Executes an OpenGL rendering operation for the visual environment.
    glTexCoord2f(0, 0); glVertex2f(x,      y     )  # Executes an OpenGL rendering operation for the visual environment.
    glTexCoord2f(1, 0); glVertex2f(x + tw, y     )  # Executes an OpenGL rendering operation for the visual environment.
    glTexCoord2f(1, 1); glVertex2f(x + tw, y + th)  # Executes an OpenGL rendering operation for the visual environment.
    glTexCoord2f(0, 1); glVertex2f(x,      y + th)  # Executes an OpenGL rendering operation for the visual environment.
    glEnd()                                     # Executes an OpenGL rendering operation for the visual environment.
    glDisable(GL_TEXTURE_2D)                    # Executes an OpenGL rendering operation for the visual environment.
    glDeleteTextures([tex])                     # Executes an OpenGL rendering operation for the visual environment.
    return th                                   # Returns the computed value to the caller.


# ── Private 3D scene drawing helpers ──────────────────────────────────────

def _gl_box(w: float, h: float, d: float,       # Defines the _gl_box function used as a reusable operation in this file.
            r: float, g: float, b: float, a: float = 1.0) -> None:  # Assigns r: float, g: float, b: float, a: float for use by the following logic.
    """
    Draw a face-shaded solid box centred at the origin.
    w/h/d = full dimensions along X/Y/Z.
    r,g,b,a in [0,1].  Top face is brightest; bottom is darkest.
    Pass a < 1.0 for transparent boxes (blending must be enabled).
    """
    hx, hy, hz = w * 0.5, h * 0.5, d * 0.5      # Assigns hx, hy, hz for use by the following logic.

    glBegin(GL_QUADS)                           # Executes an OpenGL rendering operation for the visual environment.

    # Top (+Y) — brightest
    glColor4f(min(r * 1.25, 1.0), min(g * 1.25, 1.0), min(b * 1.25, 1.0), a)  # Executes an OpenGL rendering operation for the visual environment.
    glVertex3f(-hx, hy, -hz); glVertex3f(hx, hy, -hz)  # Executes an OpenGL rendering operation for the visual environment.
    glVertex3f(hx, hy,  hz); glVertex3f(-hx, hy,  hz)  # Executes an OpenGL rendering operation for the visual environment.

    # Front (−Z)
    glColor4f(r, g, b, a)                       # Executes an OpenGL rendering operation for the visual environment.
    glVertex3f(-hx, -hy, -hz); glVertex3f(hx, -hy, -hz)  # Executes an OpenGL rendering operation for the visual environment.
    glVertex3f(hx,  hy, -hz); glVertex3f(-hx,  hy, -hz)  # Executes an OpenGL rendering operation for the visual environment.

    # Back (+Z)
    glColor4f(r * 0.80, g * 0.80, b * 0.80, a)  # Executes an OpenGL rendering operation for the visual environment.
    glVertex3f( hx, -hy, hz); glVertex3f(-hx, -hy, hz)  # Executes an OpenGL rendering operation for the visual environment.
    glVertex3f(-hx,  hy, hz); glVertex3f( hx,  hy, hz)  # Executes an OpenGL rendering operation for the visual environment.

    # Right (+X)
    glColor4f(r * 0.90, g * 0.90, b * 0.90, a)  # Executes an OpenGL rendering operation for the visual environment.
    glVertex3f(hx, -hy, -hz); glVertex3f(hx, -hy, hz)  # Executes an OpenGL rendering operation for the visual environment.
    glVertex3f(hx,  hy,  hz); glVertex3f(hx,  hy, -hz)  # Executes an OpenGL rendering operation for the visual environment.

    # Left (−X)
    glColor4f(r * 0.70, g * 0.70, b * 0.70, a)  # Executes an OpenGL rendering operation for the visual environment.
    glVertex3f(-hx, -hy,  hz); glVertex3f(-hx, -hy, -hz)  # Executes an OpenGL rendering operation for the visual environment.
    glVertex3f(-hx,  hy, -hz); glVertex3f(-hx,  hy,  hz)  # Executes an OpenGL rendering operation for the visual environment.

    # Bottom (−Y) — darkest
    glColor4f(r * 0.45, g * 0.45, b * 0.45, a)  # Executes an OpenGL rendering operation for the visual environment.
    glVertex3f(-hx, -hy,  hz); glVertex3f( hx, -hy,  hz)  # Executes an OpenGL rendering operation for the visual environment.
    glVertex3f( hx, -hy, -hz); glVertex3f(-hx, -hy, -hz)  # Executes an OpenGL rendering operation for the visual environment.

    glEnd()                                     # Executes an OpenGL rendering operation for the visual environment.


def _gl_draw_floor() -> None:                   # Defines the _gl_draw_floor function used as a reusable operation in this file.
    # Ground quad
    glColor3f(0.05, 0.12, 0.05)                 # Executes an OpenGL rendering operation for the visual environment.
    glBegin(GL_QUADS)                           # Executes an OpenGL rendering operation for the visual environment.
    glVertex3f(-HALF, 0.0,  -HALF); glVertex3f(HALF, 0.0, -HALF)  # Executes an OpenGL rendering operation for the visual environment.
    glVertex3f( HALF, 0.0,   HALF); glVertex3f(-HALF, 0.0,  HALF)  # Executes an OpenGL rendering operation for the visual environment.
    glEnd()                                     # Executes an OpenGL rendering operation for the visual environment.

    # Grid lines
    glColor3f(0.10, 0.20, 0.10)                 # Executes an OpenGL rendering operation for the visual environment.
    glBegin(GL_LINES)                           # Executes an OpenGL rendering operation for the visual environment.
    for i in range(-10, 11):                    # Starts a loop that repeats the indented logic for each item in the sequence.
        v = float(i * 10)                       # Assigns v for use by the following logic.
        glVertex3f(v,    0.05, -HALF); glVertex3f(v,    0.05, HALF)  # Executes an OpenGL rendering operation for the visual environment.
        glVertex3f(-HALF, 0.05, v   ); glVertex3f(HALF, 0.05, v   )  # Executes an OpenGL rendering operation for the visual environment.
    glEnd()                                     # Executes an OpenGL rendering operation for the visual environment.

    # Arena border
    glLineWidth(2.0)                            # Executes an OpenGL rendering operation for the visual environment.
    glColor3f(0.25, 0.50, 0.25)                 # Executes an OpenGL rendering operation for the visual environment.
    glBegin(GL_LINE_LOOP)                       # Executes an OpenGL rendering operation for the visual environment.
    glVertex3f(-HALF, 0.1, -HALF); glVertex3f(HALF, 0.1, -HALF)  # Executes an OpenGL rendering operation for the visual environment.
    glVertex3f( HALF, 0.1,  HALF); glVertex3f(-HALF, 0.1,  HALF)  # Executes an OpenGL rendering operation for the visual environment.
    glEnd()                                     # Executes an OpenGL rendering operation for the visual environment.
    glLineWidth(1.0)                            # Executes an OpenGL rendering operation for the visual environment.


def _gl_draw_trees(trees: list) -> None:        # Defines the _gl_draw_trees function used as a reusable operation in this file.
    for t in trees:                             # Starts a loop that repeats the indented logic for each item in the sequence.
        glPushMatrix()                          # Executes an OpenGL rendering operation for the visual environment.
        glTranslatef(t["x"], 7.5, t["z"])   # tree height 15 → centre at 7.5  # Presentation note: Executes an OpenGL rendering operation for the visual environment.
        _gl_box(TREE_R * 2, 15.0, TREE_R * 2, 0.08, 0.32, 0.08)  # Executes one step of the file logic needed by the environment, demo, or test workflow.
        glPopMatrix()                           # Executes an OpenGL rendering operation for the visual environment.


_VEH_COL_GL = {                                 # Defines constant _VEH_COL_GL used to control environment behavior consistently.
    "red":         (0.86, 0.20, 0.20),  "lime":        (0.20, 0.86, 0.20),  # Continues the current multi-line collection or argument list.
    "cyan":        (0.20, 0.78, 0.86),  "orange":      (0.86, 0.55, 0.20),  # Continues the current multi-line collection or argument list.
    "magenta":     (0.86, 0.20, 0.86),  "yellow":      (0.86, 0.86, 0.20),  # Continues the current multi-line collection or argument list.
    "deepskyblue": (0.00, 0.71, 1.00),  "tomato":      (1.00, 0.39, 0.28),  # Continues the current multi-line collection or argument list.
    "chartreuse":  (0.50, 1.00, 0.00),  "gold":        (1.00, 0.84, 0.00),  # Continues the current multi-line collection or argument list.
    "hotpink":     (1.00, 0.41, 0.71),  "turquoise":   (0.25, 0.88, 0.82),  # Continues the current multi-line collection or argument list.
    "coral":       (1.00, 0.50, 0.31),  "violet":      (0.93, 0.51, 0.93),  # Continues the current multi-line collection or argument list.
}                                               # Executes one step of the file logic needed by the environment, demo, or test workflow.
_DEFAULT_COL_GL = (0.60, 0.60, 0.60)            # Defines constant _DEFAULT_COL_GL used to control environment behavior consistently.


def _gl_draw_vehicles(vehicles: list) -> None:  # Defines the _gl_draw_vehicles function used as a reusable operation in this file.
    for v in vehicles:                          # Starts a loop that repeats the indented logic for each item in the sequence.
        r, g, b = _VEH_COL_GL.get(v["color"], _DEFAULT_COL_GL)  # Assigns r, g, b for use by the following logic.
        if v["alive"]:                          # Starts a conditional branch that executes only when the stated condition is true.
            # Altitude shadow on ground
            glPushMatrix()                      # Executes an OpenGL rendering operation for the visual environment.
            glTranslatef(v["x"], 0.08, v["z"])  # Executes an OpenGL rendering operation for the visual environment.
            glColor4f(0.0, 0.0, 0.0, 0.35)      # Executes an OpenGL rendering operation for the visual environment.
            glBegin(GL_TRIANGLE_FAN)            # Executes an OpenGL rendering operation for the visual environment.
            glVertex3f(0, 0, 0)                 # Executes an OpenGL rendering operation for the visual environment.
            for k in range(13):                 # Starts a loop that repeats the indented logic for each item in the sequence.
                a = 2 * math.pi * k / 12        # Assigns a for use by the following logic.
                glVertex3f(math.cos(a) * 2.0, 0, math.sin(a) * 2.0)  # Uses trigonometry to convert angles into direction or normalized feature values.
            glEnd()                             # Executes an OpenGL rendering operation for the visual environment.
            glPopMatrix()                       # Executes an OpenGL rendering operation for the visual environment.

            # Altitude wire (thin vertical line from ground to vehicle)
            glLineWidth(1.0)                    # Executes an OpenGL rendering operation for the visual environment.
            glColor4f(r, g, b, 0.35)            # Executes an OpenGL rendering operation for the visual environment.
            glBegin(GL_LINES)                   # Executes an OpenGL rendering operation for the visual environment.
            glVertex3f(v["x"], 0.1,    v["z"])  # Executes an OpenGL rendering operation for the visual environment.
            glVertex3f(v["x"], v["y"], v["z"])  # Executes an OpenGL rendering operation for the visual environment.
            glEnd()                             # Executes an OpenGL rendering operation for the visual environment.

            # Vehicle body — drone-like flat box oriented to heading
            glPushMatrix()                      # Executes an OpenGL rendering operation for the visual environment.
            glTranslatef(v["x"], v["y"], v["z"])  # Executes an OpenGL rendering operation for the visual environment.
            glRotatef(math.degrees(v["angle"]) + 90.0, 0, 1, 0)  # Executes an OpenGL rendering operation for the visual environment.
            _gl_box(4.0, 1.2, 3.0, r, g, b)     # Executes one step of the file logic needed by the environment, demo, or test workflow.
            # Front marker
            glColor3f(min(r * 1.6, 1.0), min(g * 1.6, 1.0), min(b * 1.6, 1.0))  # Executes an OpenGL rendering operation for the visual environment.
            glBegin(GL_QUADS)                   # Executes an OpenGL rendering operation for the visual environment.
            glVertex3f(-0.4, 0.62, -1.7); glVertex3f(0.4, 0.62, -1.7)  # Executes an OpenGL rendering operation for the visual environment.
            glVertex3f( 0.4, 0.62, -1.2); glVertex3f(-0.4, 0.62, -1.2)  # Executes an OpenGL rendering operation for the visual environment.
            glEnd()                             # Executes an OpenGL rendering operation for the visual environment.
            glPopMatrix()                       # Executes an OpenGL rendering operation for the visual environment.

        else:                                   # Starts the fallback branch when earlier conditions were false.
            # Ghost of dead/respawning vehicle — semi-transparent dark box
            glPushMatrix()                      # Executes an OpenGL rendering operation for the visual environment.
            glTranslatef(v["x"], v["y"], v["z"])  # Executes an OpenGL rendering operation for the visual environment.
            _gl_box(4.0, 1.2, 3.0, r * 0.30, g * 0.30, b * 0.30, a=0.30)  # Assigns _gl_box(4.0, 1.2, 3.0, r * 0.30, g * 0.30, b * 0.30, a for use by the following logic.
            glPopMatrix()                       # Executes an OpenGL rendering operation for the visual environment.


def _gl_draw_bullets(bullets: list) -> None:    # Defines the _gl_draw_bullets function used as a reusable operation in this file.
    # Tracer line (fades from bright to transparent)
    glLineWidth(2.0)                            # Executes an OpenGL rendering operation for the visual environment.
    glBegin(GL_LINES)                           # Executes an OpenGL rendering operation for the visual environment.
    for b in bullets:                           # Starts a loop that repeats the indented logic for each item in the sequence.
        glColor4f(1.0, 0.85, 0.15, 1.0)         # Executes an OpenGL rendering operation for the visual environment.
        glVertex3f(b["x"], b["y"], b["z"])      # Executes an OpenGL rendering operation for the visual environment.
        glColor4f(1.0, 0.40, 0.05, 0.0)         # Executes an OpenGL rendering operation for the visual environment.
        trail = 5.0                             # Assigns trail for use by the following logic.
        glVertex3f(b["x"] - b["dx"] * trail,    # Continues the current multi-line collection or argument list.
                   b["y"] - b["dy"] * trail,    # Continues the current multi-line collection or argument list.
                   b["z"] - b["dz"] * trail)    # Executes one step of the file logic needed by the environment, demo, or test workflow.
    glEnd()                                     # Executes an OpenGL rendering operation for the visual environment.
    glLineWidth(1.0)                            # Executes an OpenGL rendering operation for the visual environment.

    # Bright point at bullet head
    glPointSize(5.0)                            # Executes an OpenGL rendering operation for the visual environment.
    glBegin(GL_POINTS)                          # Executes an OpenGL rendering operation for the visual environment.
    for b in bullets:                           # Starts a loop that repeats the indented logic for each item in the sequence.
        glColor3f(1.0, 1.0, 0.4)                # Executes an OpenGL rendering operation for the visual environment.
        glVertex3f(b["x"], b["y"], b["z"])      # Executes an OpenGL rendering operation for the visual environment.
    glEnd()                                     # Executes an OpenGL rendering operation for the visual environment.
    glPointSize(1.0)                            # Executes an OpenGL rendering operation for the visual environment.


def _gl_draw_hunter(yaw: float, pitch: float) -> None:  # Defines the _gl_draw_hunter function used as a reusable operation in this file.
    # Turret base (sits on floor, centre at y=1.0 → occupies 0..2)
    glPushMatrix()                              # Executes an OpenGL rendering operation for the visual environment.
    glTranslatef(0.0, 1.0, 0.0)                 # Executes an OpenGL rendering operation for the visual environment.
    _gl_box(5.0, 2.0, 5.0, 0.10, 0.70, 0.20)    # Executes one step of the file logic needed by the environment, demo, or test workflow.
    glPopMatrix()                               # Executes an OpenGL rendering operation for the visual environment.

    # Barrel direction vector
    cp = math.cos(pitch)                        # Uses trigonometry to convert angles into direction or normalized feature values.
    dx = math.cos(yaw) * cp                     # Uses trigonometry to convert angles into direction or normalized feature values.
    dy = math.sin(pitch)                        # Uses trigonometry to convert angles into direction or normalized feature values.
    dz = math.sin(yaw) * cp                     # Uses trigonometry to convert angles into direction or normalized feature values.
    blen = 18.0                                 # Assigns blen for use by the following logic.

    # Barrel line
    glLineWidth(4.0)                            # Executes an OpenGL rendering operation for the visual environment.
    glColor3f(0.20, 1.00, 0.40)                 # Executes an OpenGL rendering operation for the visual environment.
    glBegin(GL_LINES)                           # Executes an OpenGL rendering operation for the visual environment.
    glVertex3f(0.0, 2.3, 0.0)                   # Executes an OpenGL rendering operation for the visual environment.
    glVertex3f(dx * blen, 2.3 + dy * blen, dz * blen)  # Executes an OpenGL rendering operation for the visual environment.
    glEnd()                                     # Executes an OpenGL rendering operation for the visual environment.

    # Faint aim ray extending into the distance (disable depth test so it
    # is never occluded by terrain or vehicles — it should always be visible)
    glDisable(GL_DEPTH_TEST)                    # Executes an OpenGL rendering operation for the visual environment.
    glLineWidth(1.0)                            # Executes an OpenGL rendering operation for the visual environment.
    glColor4f(0.20, 1.00, 0.40, 0.12)           # Executes an OpenGL rendering operation for the visual environment.
    glBegin(GL_LINES)                           # Executes an OpenGL rendering operation for the visual environment.
    glVertex3f(0.0, 2.3, 0.0)                   # Executes an OpenGL rendering operation for the visual environment.
    glVertex3f(dx * 120.0, 2.3 + dy * 120.0, dz * 120.0)  # Executes an OpenGL rendering operation for the visual environment.
    glEnd()                                     # Executes an OpenGL rendering operation for the visual environment.
    glEnable(GL_DEPTH_TEST)                     # Executes an OpenGL rendering operation for the visual environment.


def _gl_draw_hud(s: dict, hud_font, W: int) -> None:  # Defines the _gl_draw_hud function used as a reusable operation in this file.
    """Built-in stats HUD drawn as a 2D overlay in the top-left corner."""
    yaw      = s["hunterYaw"]                   # Assigns yaw for use by the following logic.
    pitch    = s["hunterPitch"]                 # Assigns pitch for use by the following logic.
    alive_ai = sum(1 for v in s["vehicles"] if v["isAI"] and v["alive"])  # Assigns alive_ai for use by the following logic.

    lines = [                                   # Assigns lines for use by the following logic.
        (f"Tick  : {s['tick']:5d} / {MAX_GAME_TICKS}", (160, 190, 160)),  # Continues a multi-line collection, tuple, or function argument list.
        (f"Time  : {s['tick'] / TICK_RATE:.1f} s",     (160, 190, 160)),  # Continues a multi-line collection, tuple, or function argument list.
        (f"Score : {s['hunterScore']}",                  (110, 255, 110)),  # Continues a multi-line collection, tuple, or function argument list.
        (f"AI    : {alive_ai} alive  "          # Continues a multi-line collection, tuple, or function argument list.
         f"wave {s['waves_launched']}/{NUM_PERIODIC_WAVES}  "  # Executes one step of the file logic needed by the environment, demo, or test workflow.
         f"cap {s['wave_capacity']}",                    (160, 190, 160)),  # Continues the current multi-line collection or argument list.
        (f"Shots : {len(s['bullets'])} in flight",       (160, 190, 160)),  # Continues a multi-line collection, tuple, or function argument list.
        (f"Yaw {math.degrees(yaw):+.1f}°  Pitch {math.degrees(pitch):+.1f}°",  # Continues a multi-line collection, tuple, or function argument list.
         (200, 200, 140)),                      # Continues a multi-line collection, tuple, or function argument list.
    ]                                           # Executes one step of the file logic needed by the environment, demo, or test workflow.
    if s["gameOver"]:                           # Starts a conditional branch that executes only when the stated condition is true.
        lines.insert(0, (f">>> {s['gameOverReason']}", (255, 80, 80)))  # Executes one step of the file logic needed by the environment, demo, or test workflow.

    LINE_H  = 18                                # Defines constant LINE_H used to control environment behavior consistently.
    PAD     = 6                                 # Defines constant PAD used to control environment behavior consistently.
    panel_w = 290                               # Assigns panel_w for use by the following logic.
    panel_h = len(lines) * LINE_H + PAD         # Assigns panel_h for use by the following logic.

    gl_begin_2d(W, W)                           # Switches OpenGL into 2D overlay mode for screen-space drawing.
    gl_fill_rect(0, W - panel_h, panel_w, panel_h, 0.02, 0.02, 0.02, 0.65)  # Draws a filled rectangle used for UI panels, dividers, or background shapes.
    for i, (text, color) in enumerate(lines):   # Starts a loop that repeats the indented logic for each item in the sequence.
        y = W - PAD - (i + 1) * LINE_H          # Assigns y for use by the following logic.
        gl_blit(text, PAD, y, hud_font, color)  # Draws text on the OpenGL surface using a pygame-rendered texture.
    gl_end_2d()                                 # Restores OpenGL state after drawing the 2D overlay.


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — GAME ENGINE  (pure Python — no Gymnasium dependency)
#
#  INPUT  → action : int  (one of NUM_ACTIONS discrete choices)
#  OUTPUT → (obs, reward, terminated, truncated, info)
# ═══════════════════════════════════════════════════════════════════════════

class GameEngine:                               # Defines the pure game simulation layer that contains physics, spawning, rewards, and game state.
    """
    Self-contained 3D game engine.  Gymnasium-agnostic.

    Parameters
    ----------
    rng : np.random.Generator
        Seeded random-number generator (pass env.np_random from ShooterEnv).

    Typical usage
    -------------
        engine = GameEngine(rng)
        engine.reset()
        obs = engine.get_obs()                    # float32[169]
        obs, reward, terminated, truncated, info = engine.step(action)
    """

    def __init__(self, rng: np.random.Generator):  # Initializes the object and stores setup values required by later method calls.
        self._rng  = rng                        # Stores self._rng as object state so other methods can reuse it later.
        self.state: dict = {}                   # Stores self.state: dict as object state so other methods can reuse it later.

    # ── Public interface ───────────────────────────────────────────────────

    def reset(self) -> None:                    # Resets the simulation or environment to a fresh episode state.
        self.state    = self._build_initial_state()  # Stores self.state as object state so other methods can reuse it later.
        trees         = self.state["trees"]     # Stores trees as object state so other methods can reuse it later.
        self._tree_xs = np.array([t["x"] for t in trees], dtype=np.float32)  # Converts Python values into a NumPy array for vectorized numerical operations.
        self._tree_zs = np.array([t["z"] for t in trees], dtype=np.float32)  # Converts Python values into a NumPy array for vectorized numerical operations.

    def step(self, action: int):                # Advances the environment by one action and returns the standard DRL transition tuple.
        """
        Advance the simulation by one tick.

        INPUT
        -----
        action : int ∈ {0 … NUM_ACTIONS−1}

        OUTPUT
        ------
        obs        : np.ndarray float32 shape (OBS_SIZE,)
        reward     : float
        terminated : bool
        truncated  : bool
        info       : dict  — tick, hunterScore, alive_ai, total_ai,
                             bullets_in_flight, gameOverReason
        """
        prev_alive = sum(1 for v in self.state["vehicles"] if v["alive"])  # Stores prev_alive as object state so other methods can reuse it later.

        self._apply_action(action)              # Executes one step of the file logic needed by the environment, demo, or test workflow.
        self._tick()                            # Executes one step of the file logic needed by the environment, demo, or test workflow.

        curr_alive = sum(1 for v in self.state["vehicles"] if v["alive"])  # Stores curr_alive as object state so other methods can reuse it later.
        obs        = self.get_obs()             # Stores obs as object state so other methods can reuse it later.
        reward     = self._compute_reward(action, prev_alive, curr_alive)  # Stores reward as object state so other methods can reuse it later.
        terminated = bool(self.state["gameOver"])  # Stores terminated as object state so other methods can reuse it later.
        truncated  = self.state["tick"] >= MAX_GAME_TICKS and not terminated  # Stores truncated as object state so other methods can reuse it later.
        info       = self._build_info()         # Stores info as object state so other methods can reuse it later.

        return obs, reward, terminated, truncated, info  # Returns the computed value to the caller.

    def get_obs(self) -> np.ndarray:            # Builds the normalized observation vector that becomes the neural-network input.
        """
        Build the 169-element normalised observation vector.

        Feature blocks
        ──────────────
        [0  … 2 ]  global   : tick/4800, score/500, alive_count/15 (MAX_VEH_SLOTS)
        [3  … 5 ]  turret   : yaw/π, pitch/0.9, roll/0.25
        [6  …125]  vehicles : 15 slots × 8 values (nearest-xz first, zero-padded)
                              x/100, z/100, dist_xz/141, sinθ, cosθ, y/20, alive, isAI
        [126…165]  bullets  : 10 slots × 4 values (zero-padded)
                              x/100, z/100, dx, dz
        [166…168]  threat   : 3d_dist/141, xz_angle/π, approach-speed proxy
        """
        s   = self.state                        # Stores s as object state so other methods can reuse it later.
        obs = np.zeros(OBS_SIZE, dtype=np.float32)  # Creates a zero-filled NumPy array used as a fixed-size observation buffer.

        # — Global state (indices 0–2) ────────────────────────────────────
        obs[0] = s["tick"]        / MAX_GAME_TICKS  # Assigns obs[0] for use by the following logic.
        obs[1] = s["hunterScore"] / 500.0       # Assigns obs[1] for use by the following logic.
        obs[2] = sum(1 for v in s["vehicles"] if v["alive"]) / MAX_VEH_SLOTS  # Assigns obs[2] for use by the following logic.

        # — Turret state (indices 3–5) ────────────────────────────────────
        obs[3] = s["hunterYaw"]   / math.pi     # Assigns obs[3] for use by the following logic.
        obs[4] = s["hunterPitch"] / 0.9         # Assigns obs[4] for use by the following logic.
        obs[5] = s["hunterRoll"]  / 0.25        # Assigns obs[5] for use by the following logic.

        # — Vehicle features (indices 6–125) — sort by squared XZ distance ─
        vehs = sorted(s["vehicles"], key=lambda v: v["x"] ** 2 + v["z"] ** 2)  # Assigns vehs for use by the following logic.
        for i in range(min(len(vehs), MAX_VEH_SLOTS)):  # Starts a loop that repeats the indented logic for each item in the sequence.
            v     = vehs[i]                     # Assigns v for use by the following logic.
            d_xz  = math.sqrt(v["x"] ** 2 + v["z"] ** 2)  # Computes Euclidean distance from squared coordinate differences.
            angle = math.atan2(v["z"], v["x"])  # Computes an angle from vector components, useful for yaw, pitch, and direction logic.
            j = OBS.VEH_START + i * 8           # Assigns j for use by the following logic.
            obs[j]     = v["x"] / 100.0         # Assigns obs[j] for use by the following logic.
            obs[j + 1] = v["z"] / 100.0         # Assigns obs[j + 1] for use by the following logic.
            obs[j + 2] = d_xz   / 141.0         # Assigns obs[j + 2] for use by the following logic.
            obs[j + 3] = math.sin(angle)        # Uses trigonometry to convert angles into direction or normalized feature values.
            obs[j + 4] = math.cos(angle)        # Uses trigonometry to convert angles into direction or normalized feature values.
            obs[j + 5] = v["y"] / 20.0          # Assigns obs[j + 5] for use by the following logic.
            obs[j + 6] = 1.0 if v["alive"] else 0.0  # Assigns obs[j + 6] for use by the following logic.
            obs[j + 7] = 1.0 if v["isAI"]  else 0.0  # Assigns obs[j + 7] for use by the following logic.

        # — Bullet features (indices 126–165) ─────────────────────────────
        for i in range(min(len(s["bullets"]), MAX_BULLET_SLOTS)):  # Starts a loop that repeats the indented logic for each item in the sequence.
            b = s["bullets"][i]                 # Assigns b for use by the following logic.
            j = OBS.BULLET_START + i * 4        # Assigns j for use by the following logic.
            obs[j]     = b["x"]  / 100.0        # Assigns obs[j] for use by the following logic.
            obs[j + 1] = b["z"]  / 100.0        # Assigns obs[j + 1] for use by the following logic.
            obs[j + 2] = b["dx"]                # Assigns obs[j + 2] for use by the following logic.
            obs[j + 3] = b["dz"]                # Assigns obs[j + 3] for use by the following logic.

        # — Nearest-threat summary (indices 166–168) ──────────────────────
        alive = [v for v in s["vehicles"] if v["alive"]]  # Assigns alive for use by the following logic.
        base  = OBS.THREAT_START                # Assigns base for use by the following logic.
        if alive:                               # Starts a conditional branch that executes only when the stated condition is true.
            n        = min(alive, key=lambda v: v["x"] ** 2 + v["z"] ** 2)  # Assigns n for use by the following logic.
            nd3      = _dist3(n["x"], n["y"], n["z"], 0.0, 2.3, 0.0)  # Assigns nd3 for use by the following logic.
            approach = n["x"] * math.cos(n["angle"]) + n["z"] * math.sin(n["angle"])  # Uses trigonometry to convert angles into direction or normalized feature values.
            obs[base]     = nd3 / 141.0         # Assigns obs[base] for use by the following logic.
            obs[base + 1] = math.atan2(n["z"], n["x"]) / math.pi  # Computes an angle from vector components, useful for yaw, pitch, and direction logic.
            obs[base + 2] = approach            # Assigns obs[base + 2] for use by the following logic.
        else:                                   # Starts the fallback branch when earlier conditions were false.
            obs[base] = 1.0                     # Assigns obs[base] for use by the following logic.

        return obs                              # Returns the computed value to the caller.

    # ── State initialisation ───────────────────────────────────────────────

    def _build_initial_state(self) -> dict:     # Defines the _build_initial_state function used as a reusable operation in this file.
        trees        = _gen_trees(self._rng)    # Stores trees as object state so other methods can reuse it later.
        vehicles     = []                       # Assigns vehicles for use by the following logic.
        angle_offset = float(self._rng.uniform(0, 2 * math.pi))  # Stores angle_offset as object state so other methods can reuse it later.

        for i, color in enumerate(_AI_COLORS):  # Starts a loop that repeats the indented logic for each item in the sequence.
            angle = (i / 4) * math.pi * 2 + angle_offset  # Assigns angle for use by the following logic.
            vehicles.append({                   # Executes one step of the file logic needed by the environment, demo, or test workflow.
                "id":           f"ai_{i}",      # Continues the current multi-line collection or argument list.
                "color":        color,          # Continues the current multi-line collection or argument list.
                "x":            math.cos(angle) * (HALF - 10),  # Uses trigonometry to convert angles into direction or normalized feature values.
                "z":            math.sin(angle) * (HALF - 10),  # Uses trigonometry to convert angles into direction or normalized feature values.
                "y":            float(self._rng.uniform(VEHICLE_HEIGHT_MIN,  # Continues the current multi-line collection or argument list.
                                                        VEHICLE_HEIGHT_MAX)),  # Continues the current multi-line collection or argument list.
                "angle":        math.atan2(-math.sin(angle), -math.cos(angle)),  # Computes an angle from vector components, useful for yaw, pitch, and direction logic.
                "alive":        True,           # Continues the current multi-line collection or argument list.
                "respawnTimer": 0,              # Continues the current multi-line collection or argument list.
                "isAI":         True,           # Continues the current multi-line collection or argument list.
                "stuckCounter": 0,              # Continues the current multi-line collection or argument list.
                "lastX":        0.0,            # Continues the current multi-line collection or argument list.
                "lastZ":        0.0,            # Continues the current multi-line collection or argument list.
            })                                  # Executes one step of the file logic needed by the environment, demo, or test workflow.

        return {                                # Returns the computed value to the caller.
            "tick":             0,              # Continues the current multi-line collection or argument list.
            "gameOver":         False,          # Continues the current multi-line collection or argument list.
            "gameOverReason":   "",             # Continues the current multi-line collection or argument list.
            "trees":            trees,          # Continues the current multi-line collection or argument list.
            "vehicles":         vehicles,       # Continues the current multi-line collection or argument list.
            "bullets":          [],             # Continues the current multi-line collection or argument list.
            "hunterYaw":        0.0,            # Continues the current multi-line collection or argument list.
            "hunterPitch":      0.0,            # Continues the current multi-line collection or argument list.
            "hunterRoll":       0.0,            # Continues the current multi-line collection or argument list.
            "hunterScore":      0,              # Continues the current multi-line collection or argument list.
            "nextAIIndex":      len(_AI_COLORS),  # Continues the current multi-line collection or argument list.
            "total_spawned_ai": len(_AI_COLORS),  # unique vehicles spawned this episode  # Presentation note: Executes one step of the file logic needed by the environment, demo, or test workflow.
            "wave_capacity":    INITIAL_WAVE_SIZE, # current-wave vehicle count (grows each wave)  # Presentation note: Executes one step of the file logic needed by the environment, demo, or test workflow.
            "waves_launched":   0,                 # periodic waves fired so far (max NUM_PERIODIC_WAVES)  # Presentation note: Executes one step of the file logic needed by the environment, demo, or test workflow.
        }                                       # Executes one step of the file logic needed by the environment, demo, or test workflow.

    # ── Action application ─────────────────────────────────────────────────

    def _apply_action(self, action: int) -> None:  # Defines the _apply_action function used as a reusable operation in this file.
        s = self.state                          # Stores s as object state so other methods can reuse it later.

        if   action == 2: s["hunterYaw"]  += 0.10  # Starts a conditional branch that executes only when the stated condition is true.
        elif action == 3: s["hunterYaw"]  -= 0.10  # Starts an alternative conditional branch after a previous condition was false.
        elif action == 4: s["hunterPitch"] = min(_PITCH_MAX, s["hunterPitch"] + 0.02)  # Starts an alternative conditional branch after a previous condition was false.
        elif action == 5: s["hunterPitch"] = max(_PITCH_MIN, s["hunterPitch"] - 0.02)  # Starts an alternative conditional branch after a previous condition was false.

        s["hunterYaw"] = _wrap_yaw(s["hunterYaw"])  # Assigns s["hunterYaw"] for use by the following logic.

        # Auto-aim: snap yaw AND pitch to the nearest alive vehicle.
        # _fire is set True only when a target exists, so fire_at_nearest
        # does nothing (no bullet, no score penalty) when all drones are down.
        _fire = (action == 1)                   # Executes one step of the file logic needed by the environment, demo, or test workflow.
        if action == 6:                         # Starts a conditional branch that executes only when the stated condition is true.
            alive = [v for v in s["vehicles"] if v["alive"]]  # Assigns alive for use by the following logic.
            if alive:                           # Starts a conditional branch that executes only when the stated condition is true.
                n      = min(alive, key=lambda v: _dist(v["x"], v["z"], 0.0, 0.0))  # Assigns n for use by the following logic.
                xz_d   = max(_dist(n["x"], n["z"], 0.0, 0.0), 0.5)  # Assigns xz_d for use by the following logic.
                s["hunterYaw"]   = math.atan2(n["z"], n["x"])  # Computes an angle from vector components, useful for yaw, pitch, and direction logic.
                raw_pitch        = math.atan2(n["y"] - 2.3, xz_d)  # Computes an angle from vector components, useful for yaw, pitch, and direction logic.
                s["hunterPitch"] = max(_PITCH_MIN, min(_PITCH_MAX, raw_pitch))  # Assigns s["hunterPitch"] for use by the following logic.
                _fire = True                    # Assigns _fire for use by the following logic.

        # Fire: spawn bullet from barrel tip in 3D aim direction
        if _fire:                               # Starts a conditional branch that executes only when the stated condition is true.
            s["hunterScore"] -= 2               # Assigns s["hunterScore"] - for use by the following logic.
            yaw, pitch = s["hunterYaw"], s["hunterPitch"]  # Assigns yaw, pitch for use by the following logic.
            cp = math.cos(pitch)                # Uses trigonometry to convert angles into direction or normalized feature values.
            dx = math.cos(yaw) * cp             # Uses trigonometry to convert angles into direction or normalized feature values.
            dy = math.sin(pitch)                # Uses trigonometry to convert angles into direction or normalized feature values.
            dz = math.sin(yaw) * cp             # Uses trigonometry to convert angles into direction or normalized feature values.
            s["bullets"].append({               # Executes one step of the file logic needed by the environment, demo, or test workflow.
                "x": dx * 3.0,  "y": 2.3 + dy * 3.0,  "z": dz * 3.0,  # Continues the current multi-line collection or argument list.
                "dx": dx,       "dy": dy,               "dz": dz,  # Continues the current multi-line collection or argument list.
                "dist": 0.0,                    # Continues the current multi-line collection or argument list.
            })                                  # Executes one step of the file logic needed by the environment, demo, or test workflow.

    # ── Game tick ──────────────────────────────────────────────────────────

    def _tick(self) -> None:                    # Defines the _tick function used as a reusable operation in this file.
        s = self.state                          # Stores s as object state so other methods can reuse it later.
        if s["gameOver"]:                       # Starts a conditional branch that executes only when the stated condition is true.
            return                              # Exits the current function without returning a value.

        s["tick"] += 1                          # Assigns s["tick"] + for use by the following logic.

        if s["tick"] >= MAX_GAME_TICKS:         # Starts a conditional branch that executes only when the stated condition is true.
            s["gameOver"]       = True          # Assigns s["gameOver"] for use by the following logic.
            s["gameOverReason"] = "Time is up! Hunter survived 4 minutes!"  # Assigns s["gameOverReason"] for use by the following logic.
            return                              # Exits the current function without returning a value.

        # Wave spawning — every WAVE_INTERVAL ticks, spawn int(capacity × WAVE_GROWTH)
        # additional vehicles so the field grows by ~50% each minute.
        # Waves fire at ticks 1200, 2400, 3600  (minutes 1, 2, 3).
        if (s["tick"] % WAVE_INTERVAL == 0      # Starts a conditional branch that executes only when the stated condition is true.
                and s["waves_launched"] < NUM_PERIODIC_WAVES):  # Executes one step of the file logic needed by the environment, demo, or test workflow.
            new_capacity = int(s["wave_capacity"] * WAVE_GROWTH)  # Assigns new_capacity for use by the following logic.
            to_spawn     = new_capacity - s["wave_capacity"]  # Assigns to_spawn for use by the following logic.
            for _ in range(to_spawn):           # Starts a loop that repeats the indented logic for each item in the sequence.
                ci = (s["nextAIIndex"] - len(_AI_COLORS)) % len(_EXTRA_COLORS)  # Assigns ci for use by the following logic.
                a  = float(self._rng.uniform(0, 2 * math.pi))  # Stores a as object state so other methods can reuse it later.
                s["vehicles"].append({          # Executes one step of the file logic needed by the environment, demo, or test workflow.
                    "id":           f"ai_{s['nextAIIndex']}",  # Continues the current multi-line collection or argument list.
                    "color":        _EXTRA_COLORS[ci],  # Continues the current multi-line collection or argument list.
                    "x":            math.cos(a) * (HALF - 10),  # Uses trigonometry to convert angles into direction or normalized feature values.
                    "z":            math.sin(a) * (HALF - 10),  # Uses trigonometry to convert angles into direction or normalized feature values.
                    "y":            float(self._rng.uniform(VEHICLE_HEIGHT_MIN,  # Continues the current multi-line collection or argument list.
                                                            VEHICLE_HEIGHT_MAX)),  # Continues the current multi-line collection or argument list.
                    "angle":        math.atan2(-math.sin(a), -math.cos(a)),  # Computes an angle from vector components, useful for yaw, pitch, and direction logic.
                    "alive":        True,       # Continues the current multi-line collection or argument list.
                    "respawnTimer": 0,          # Continues the current multi-line collection or argument list.
                    "isAI":         True,       # Continues the current multi-line collection or argument list.
                    "stuckCounter": 0,          # Continues the current multi-line collection or argument list.
                    "lastX":        0.0,        # Continues the current multi-line collection or argument list.
                    "lastZ":        0.0,        # Continues the current multi-line collection or argument list.
                })                              # Executes one step of the file logic needed by the environment, demo, or test workflow.
                s["nextAIIndex"]      += 1      # Assigns s["nextAIIndex"]      + for use by the following logic.
                s["total_spawned_ai"] += 1      # Assigns s["total_spawned_ai"] + for use by the following logic.
            s["wave_capacity"]  = new_capacity  # Assigns s["wave_capacity"] for use by the following logic.
            s["waves_launched"] += 1            # Assigns s["waves_launched"] + for use by the following logic.

        # AI movement and reach check
        for v in s["vehicles"]:                 # Starts a loop that repeats the indented logic for each item in the sequence.
            if s["gameOver"]:                   # Starts a conditional branch that executes only when the stated condition is true.
                break                           # Stops the nearest loop immediately.
            if v["alive"]:                      # Starts a conditional branch that executes only when the stated condition is true.
                _steer_ai(v, self._tree_xs, self._tree_zs, s["vehicles"], self._rng)  # Executes one step of the file logic needed by the environment, demo, or test workflow.
                if _dist(v["x"], v["z"], 0.0, 0.0) < REACH_DIST:  # Starts a conditional branch that executes only when the stated condition is true.
                    s["gameOver"]       = True  # Assigns s["gameOver"] for use by the following logic.
                    s["gameOverReason"] = (     # Assigns s["gameOverReason"] for use by the following logic.
                        f"{v['color'].upper()} vehicle reached the Hunter!"  # Executes one step of the file logic needed by the environment, demo, or test workflow.
                    )                           # Closes the current multi-line function call or data structure.
            else:                               # Starts the fallback branch when earlier conditions were false.
                v["respawnTimer"] -= 1          # Assigns v["respawnTimer"] - for use by the following logic.
                if v["respawnTimer"] <= 0:      # Starts a conditional branch that executes only when the stated condition is true.
                    a          = float(self._rng.uniform(0, 2 * math.pi))  # Stores a as object state so other methods can reuse it later.
                    v["x"]     = math.cos(a) * (HALF - 10)  # Uses trigonometry to convert angles into direction or normalized feature values.
                    v["z"]     = math.sin(a) * (HALF - 10)  # Uses trigonometry to convert angles into direction or normalized feature values.
                    v["angle"] = math.atan2(-v["z"], -v["x"])  # Computes an angle from vector components, useful for yaw, pitch, and direction logic.
                    v["alive"] = True           # Assigns v["alive"] for use by the following logic.
                    v["stuckCounter"] = 0       # Assigns v["stuckCounter"] for use by the following logic.
                    # Y is preserved — vehicle keeps its altitude through respawn

        if s["gameOver"]:                       # Starts a conditional branch that executes only when the stated condition is true.
            return                              # Exits the current function without returning a value.

        # Bullet physics + 3D hit detection
        surviving = []                          # Assigns surviving for use by the following logic.
        for b in s["bullets"]:                  # Starts a loop that repeats the indented logic for each item in the sequence.
            b["x"]    += b["dx"] * BULLET_SPEED  # Assigns b["x"]    + for use by the following logic.
            b["y"]    += b["dy"] * BULLET_SPEED  # Assigns b["y"]    + for use by the following logic.
            b["z"]    += b["dz"] * BULLET_SPEED  # Assigns b["z"]    + for use by the following logic.
            b["dist"] += BULLET_SPEED           # Assigns b["dist"] + for use by the following logic.
            if b["dist"] > BULLET_MAX_DIST:               continue  # Starts a conditional branch that executes only when the stated condition is true.
            if abs(b["x"]) > HALF or abs(b["z"]) > HALF: continue  # Starts a conditional branch that executes only when the stated condition is true.
            if b["y"] < -2.0 or b["y"] > 80.0:           continue  # Starts a conditional branch that executes only when the stated condition is true.
            # Trees are solid XZ pillars — no height exemption.
            if _tree_blocked_fast(self._tree_xs, self._tree_zs, b["x"], b["z"], 0.3): continue  # Starts a conditional branch that executes only when the stated condition is true.
            # 3D hit detection — single target per shot.
            hit = False                         # Assigns hit for use by the following logic.
            for v in s["vehicles"]:             # Starts a loop that repeats the indented logic for each item in the sequence.
                if not v["alive"]:              # Starts a conditional branch that executes only when the stated condition is true.
                    continue                    # Skips the remaining loop body and moves to the next iteration.
                if _dist3(b["x"], b["y"], b["z"], v["x"], v["y"], v["z"]) < HIT_DIST:  # Starts a conditional branch that executes only when the stated condition is true.
                    v["alive"]        = False   # Assigns v["alive"] for use by the following logic.
                    v["respawnTimer"] = RESPAWN_TICKS_AI  # Assigns v["respawnTimer"] for use by the following logic.
                    s["hunterScore"] += 20      # Assigns s["hunterScore"] + for use by the following logic.
                    hit = True                  # Assigns hit for use by the following logic.
                    break                       # Stops the nearest loop immediately.
            if not hit:                         # Starts a conditional branch that executes only when the stated condition is true.
                surviving.append(b)             # Executes one step of the file logic needed by the environment, demo, or test workflow.
        s["bullets"] = surviving                # Assigns s["bullets"] for use by the following logic.

        # Win: all periodic waves launched AND all AI simultaneously down.
        # Prevents an early win from the initial 4 drones; agent must hold until
        # all 3 periodic waves have fired (minute 3) and clear the field.
        all_ai = [v for v in s["vehicles"] if v["isAI"]]  # Assigns all_ai for use by the following logic.
        if (all_ai                              # Starts a conditional branch that executes only when the stated condition is true.
                and all(not v["alive"] for v in all_ai)  # Executes one step of the file logic needed by the environment, demo, or test workflow.
                and s["waves_launched"] >= NUM_PERIODIC_WAVES  # Executes one step of the file logic needed by the environment, demo, or test workflow.
                and s["tick"] > 20):            # Executes one step of the file logic needed by the environment, demo, or test workflow.
            s["gameOver"]       = True          # Assigns s["gameOver"] for use by the following logic.
            s["gameOverReason"] = "All AI vehicles eliminated! Hunter wins!"  # Assigns s["gameOverReason"] for use by the following logic.

    # ── Reward ─────────────────────────────────────────────────────────────

    def _compute_reward(self, action: int,      # Defines the _compute_reward function used as a reusable operation in this file.
                        prev_alive: int, curr_alive: int) -> float:  # Executes one step of the file logic needed by the environment, demo, or test workflow.
        s      = self.state                     # Stores s as object state so other methods can reuse it later.
        reward = 0.0                            # Assigns reward for use by the following logic.

        reward += max(0, prev_alive - curr_alive) * 20.0  # Assigns reward + for use by the following logic.

        if action in (1, 6):                    # Starts a conditional branch that executes only when the stated condition is true.
            reward -= 2.0                       # Assigns reward - for use by the following logic.

        alive = [v for v in s["vehicles"] if v["alive"]]  # Assigns alive for use by the following logic.
        if alive:                               # Starts a conditional branch that executes only when the stated condition is true.
            nd = min(math.sqrt(v["x"] ** 2 + v["z"] ** 2) for v in alive)  # Computes Euclidean distance from squared coordinate differences.
            if nd < 20:                         # Starts a conditional branch that executes only when the stated condition is true.
                reward -= (20.0 - nd) * 0.05    # Assigns reward - for use by the following logic.

        if s["gameOver"]:                       # Starts a conditional branch that executes only when the stated condition is true.
            if "reached" in s["gameOverReason"]:  # Starts a conditional branch that executes only when the stated condition is true.
                reward -= 100.0                 # Assigns reward - for use by the following logic.
            elif "eliminated" in s["gameOverReason"]:  # Starts an alternative conditional branch after a previous condition was false.
                reward += 200.0                 # Assigns reward + for use by the following logic.
            # time-limit survival: no bonus

        return reward                           # Returns the computed value to the caller.

    # ── Info dict ──────────────────────────────────────────────────────────

    def _build_info(self) -> dict:              # Defines the _build_info function used as a reusable operation in this file.
        s = self.state                          # Stores s as object state so other methods can reuse it later.
        return {                                # Returns the computed value to the caller.
            "tick":              s["tick"],     # Continues the current multi-line collection or argument list.
            "hunterScore":       s["hunterScore"],  # Continues the current multi-line collection or argument list.
            "alive_ai":          sum(1 for v in s["vehicles"]  # Executes one step of the file logic needed by the environment, demo, or test workflow.
                                     if v["isAI"] and v["alive"]),  # Starts a conditional branch that executes only when the stated condition is true.
            "total_ai":          sum(1 for v in s["vehicles"] if v["isAI"]),  # Continues the current multi-line collection or argument list.
            "total_spawned_ai":  s["total_spawned_ai"],  # Continues the current multi-line collection or argument list.
            "wave":              s["waves_launched"],  # Continues the current multi-line collection or argument list.
            "wave_capacity":     s["wave_capacity"],  # Continues the current multi-line collection or argument list.
            "bullets_in_flight": len(s["bullets"]),  # Continues the current multi-line collection or argument list.
            "gameOverReason":    s["gameOverReason"],  # Continues the current multi-line collection or argument list.
        }                                       # Executes one step of the file logic needed by the environment, demo, or test workflow.


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — GYMNASIUM WRAPPER  (thin shell around GameEngine)
# ═══════════════════════════════════════════════════════════════════════════

class ShooterEnv(gym.Env):                      # Defines the Gymnasium wrapper used by reinforcement-learning algorithms.
    """
    Farama Gymnasium environment — thin wrapper around GameEngine.

    Inputs (agent → environment)
    ─────────────────────────────
    reset(seed, options) → (obs, info)
    step(action)         → (obs, reward, terminated, truncated, info)
      action ∈ Discrete(7)   see ACTION_NAMES

    Outputs (environment → agent)
    ──────────────────────────────
    obs        Box(169,) float32  — see OBS class for feature layout
    reward     float
    terminated bool               game-over event
    truncated  bool               tick limit reached
    info       dict               tick, hunterScore, alive_ai, total_ai,
                                  bullets_in_flight, gameOverReason

    Parameters
    ──────────
    render_mode : "human" | "rgb_array" | None

    Rendering
    ─────────
    Requires PyOpenGL when render_mode is not None:
        pip install PyOpenGL PyOpenGL_accelerate
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": TICK_RATE}  # Assigns metadata for use by the following logic.

    def __init__(self, render_mode: Optional[str] = None):  # Initializes the object and stores setup values required by later method calls.
        super().__init__()                      # Initializes behavior inherited from the parent Gymnasium environment class.

        self.observation_space = spaces.Box(    # Defines the continuous observation space expected by Gymnasium-compatible DRL code.
            low=-np.inf, high=np.inf, shape=(OBS_SIZE,), dtype=np.float32  # Assigns low for use by the following logic.
        )                                       # Closes the current multi-line function call or data structure.
        # RL agents get Discrete(6); human play adds action 6 (auto-aim)
        n_actions = NUM_ACTIONS if render_mode == "human" else NUM_ACTIONS - 1  # Executes one step of the file logic needed by the environment, demo, or test workflow.
        self.action_space = spaces.Discrete(n_actions)  # Defines the discrete action space available to the agent.

        assert render_mode is None or render_mode in self.metadata["render_modes"]  # Validates an expected condition and raises an error if the condition is false.
        self.render_mode = render_mode          # Stores self.render_mode as object state so other methods can reuse it later.

        self._engine: Optional[GameEngine] = None  # Stores self._engine: Optional[GameEngine] as object state so other methods can reuse it later.
        self._window  = None                    # Stores self._window as object state so other methods can reuse it later.
        self._clock   = None                    # Stores self._clock as object state so other methods can reuse it later.
        self._hud_font = None                   # Stores self._hud_font as object state so other methods can reuse it later.
        self._WINDOW_SIZE = 700                 # Stores self._WINDOW_SIZE as object state so other methods can reuse it later.
        # Optional callable() inserted just before pygame.display.flip().
        # Called with no arguments — use OpenGL calls or gl_begin_2d/gl_blit
        # helpers from this module to draw overlays inside the hook.
        self._pre_flip_hook = None              # Stores self._pre_flip_hook as object state so other methods can reuse it later.

    # ── Gymnasium API ──────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None,  # Resets the simulation or environment to a fresh episode state.
              options: Optional[dict] = None):  # Assigns options: Optional[dict] for use by the following logic.
        super().reset(seed=seed)                # Starts a new episode and returns the initial observation plus diagnostic info.
        self._engine = GameEngine(self.np_random)  # Stores self._engine as object state so other methods can reuse it later.
        self._engine.reset()                    # Starts a new episode and returns the initial observation plus diagnostic info.
        if self.render_mode == "human":         # Starts a conditional branch that executes only when the stated condition is true.
            self._render_frame()                # Executes one step of the file logic needed by the environment, demo, or test workflow.
        return self._engine.get_obs(), self._engine._build_info()  # Returns the computed value to the caller.

    def step(self, action: int):                # Advances the environment by one action and returns the standard DRL transition tuple.
        obs, reward, terminated, truncated, info = self._engine.step(int(action))  # Sends an action to the environment and receives the next DRL transition result.
        if self.render_mode == "human":         # Starts a conditional branch that executes only when the stated condition is true.
            self._render_frame()                # Executes one step of the file logic needed by the environment, demo, or test workflow.
        return obs, reward, terminated, truncated, info  # Returns the computed value to the caller.

    def render(self):                           # Returns or displays a rendered frame depending on the selected render mode.
        if self.render_mode == "rgb_array":     # Starts a conditional branch that executes only when the stated condition is true.
            return self._render_frame()         # Returns the computed value to the caller.

    def close(self):                            # Releases rendering resources and closes any active display window.
        if self._window is not None:            # Starts a conditional branch that executes only when the stated condition is true.
            import pygame                       # Imports pygame for keyboard input, window events, and font rendering.
            pygame.display.quit()               # Executes a pygame operation for window, keyboard, or text handling.
            pygame.quit()                       # Executes a pygame operation for window, keyboard, or text handling.
        self._window   = None                   # Stores self._window as object state so other methods can reuse it later.
        self._clock    = None                   # Stores self._clock as object state so other methods can reuse it later.
        self._hud_font = None                   # Stores self._hud_font as object state so other methods can reuse it later.
        self._engine   = None                   # Stores self._engine as object state so other methods can reuse it later.

    # ── PyOpenGL 3D renderer ───────────────────────────────────────────────

    def _render_frame(self):                    # Defines the _render_frame function used as a reusable operation in this file.
        import pygame                           # Imports pygame for keyboard input, window events, and font rendering.

        if self.render_mode is not None and not _OPENGL_OK:  # Starts a conditional branch that executes only when the stated condition is true.
            raise ImportError(                  # Re-raises or creates an error after the failure has been detected.
                "PyOpenGL is required for rendering.\n"  # Executes one step of the file logic needed by the environment, demo, or test workflow.
                "  pip install PyOpenGL PyOpenGL_accelerate"  # Executes one step of the file logic needed by the environment, demo, or test workflow.
            )                                   # Closes the current multi-line function call or data structure.

        W = self._WINDOW_SIZE                   # Stores W as object state so other methods can reuse it later.
        s = self._engine.state                  # Stores s as object state so other methods can reuse it later.

        # Initialise window + OpenGL state once
        if self._window is None:                # Starts a conditional branch that executes only when the stated condition is true.
            pygame.init()                       # Executes a pygame operation for window, keyboard, or text handling.
            pygame.display.init()               # Executes a pygame operation for window, keyboard, or text handling.
            self._window = pygame.display.set_mode(  # Stores self._window as object state so other methods can reuse it later.
                (W, W), pygame.DOUBLEBUF | pygame.OPENGL  # Continues a multi-line collection, tuple, or function argument list.
            )                                   # Closes the current multi-line function call or data structure.
            pygame.display.set_caption("Shooter-v0  |  Hunter RL Environment")  # Executes a pygame operation for window, keyboard, or text handling.
            self._clock    = pygame.time.Clock()  # Stores self._clock as object state so other methods can reuse it later.
            self._hud_font = pygame.font.SysFont("monospace", 14)  # Creates a font object used to render readable text overlays.

            glEnable(GL_DEPTH_TEST)             # Executes an OpenGL rendering operation for the visual environment.
            glEnable(GL_BLEND)                  # Executes an OpenGL rendering operation for the visual environment.
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)  # Executes an OpenGL rendering operation for the visual environment.

            glMatrixMode(GL_PROJECTION)         # Executes an OpenGL rendering operation for the visual environment.
            glLoadIdentity()                    # Executes an OpenGL rendering operation for the visual environment.
            gluPerspective(50.0, 1.0, 0.5, 1500.0)  # Executes an OpenGL rendering operation for the visual environment.
            glMatrixMode(GL_MODELVIEW)          # Executes an OpenGL rendering operation for the visual environment.

        # Clear frame
        glClearColor(0.04, 0.06, 0.12, 1.0)     # Executes an OpenGL rendering operation for the visual environment.
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # Executes an OpenGL rendering operation for the visual environment.

        # Fixed overhead-angle camera
        glLoadIdentity()                        # Executes an OpenGL rendering operation for the visual environment.
        gluLookAt(0.0, 130.0, 85.0,   # eye position  # Presentation note: Executes an OpenGL rendering operation for the visual environment.
                  0.0,   5.0,  0.0,   # look-at target  # Presentation note: Executes one step of the file logic needed by the environment, demo, or test workflow.
                  0.0,   1.0,  0.0)   # up vector  # Presentation note: Executes one step of the file logic needed by the environment, demo, or test workflow.

        # 3D scene
        _gl_draw_floor()                        # Executes one step of the file logic needed by the environment, demo, or test workflow.
        _gl_draw_trees(s["trees"])              # Executes one step of the file logic needed by the environment, demo, or test workflow.
        _gl_draw_vehicles(s["vehicles"])        # Executes one step of the file logic needed by the environment, demo, or test workflow.
        _gl_draw_bullets(s["bullets"])          # Executes one step of the file logic needed by the environment, demo, or test workflow.
        _gl_draw_hunter(s["hunterYaw"], s["hunterPitch"])  # Executes one step of the file logic needed by the environment, demo, or test workflow.

        # Built-in HUD overlay
        _gl_draw_hud(s, self._hud_font, W)      # Executes one step of the file logic needed by the environment, demo, or test workflow.

        # External overlay hook (e.g. play_shooter.py controls panel)
        if self._pre_flip_hook is not None:     # Starts a conditional branch that executes only when the stated condition is true.
            self._pre_flip_hook()               # Executes one step of the file logic needed by the environment, demo, or test workflow.

        if self.render_mode == "human":         # Starts a conditional branch that executes only when the stated condition is true.
            pygame.event.pump()                 # Executes a pygame operation for window, keyboard, or text handling.
            pygame.display.flip()               # Executes a pygame operation for window, keyboard, or text handling.
            self._clock.tick(self.metadata["render_fps"])  # Executes one step of the file logic needed by the environment, demo, or test workflow.
        else:                                   # Starts the fallback branch when earlier conditions were false.
            # rgb_array: drain the event queue so the OS doesn't mark the
            # window as unresponsive when stepping headlessly at speed
            pygame.event.pump()                 # Executes a pygame operation for window, keyboard, or text handling.
            # rgb_array: read back the OpenGL framebuffer
            glReadBuffer(GL_BACK)               # Executes an OpenGL rendering operation for the visual environment.
            data = glReadPixels(0, 0, W, W, GL_RGB, GL_UNSIGNED_BYTE)  # Assigns data for use by the following logic.
            img  = np.frombuffer(data, dtype=np.uint8).reshape(W, W, 3)  # Assigns img for use by the following logic.
            return np.flipud(img)               # Returns the computed value to the caller.
