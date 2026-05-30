"""
================================================================================
TikshiV1 — Bot Rocket League 1v1, niveau SSL, SANS spécialisation aérienne.
================================================================================

Objectif : le meilleur bot 1v1 possible.
  - Sol ULTRA dominant (dribbles, flicks, speedflip kickoff, shadow defense,
    challenge/50, recovery, gestion boost, shots).
  - Aérien EXCELLENT (aerials, wall play, ceiling, air dribbles, flip resets,
    double taps) — mais JAMAIS prioritaire sur une meilleure solution au sol.
  - Progression de phases AUTOMATIQUE (auto-pilot), aucune intervention manuelle.

Pipeline : RLGym 2.0 + RocketSim + rlgym-ppo + rlgym-tools.

Sources de design des rewards :
  - RLGym-PPO-Guide (ZealanL) : touch pondéré par la force, air-touch =
    min(air_time, height) pour de VRAIES aériennes, pas de goal reward écrasant,
    rewards zero-sum uniquement pour ce que l'adversaire doit empêcher.
  - Lucy-SKG (papier, bat Necto/Nexto) : reward shaping utilitaire.
  - Necto / Nexto : opp_punish (zero-sum), potential-based shaping.

ATTENTION : ce script S'ENTRAÎNE (GPU + RocketSim, milliards de steps). Le
niveau SSL dépend du temps de calcul investi. Lance-le simplement avec
`python TikshiV1.py` : il détecte la phase depuis les checkpoints et progresse
tout seul.
"""

import os
import json
import math
import socket
import numpy as np

from typing import Dict, Any

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG GLOBALE
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_NAME = "TikshiV1"

# Mettre "cpu" pour forcer le CPU (test sans GPU). "cuda" = GPU.
DEVICE = "cuda"

# [1V1] Taille d'équipe = 1. C'est LE changement central : bot 1v1.
TEAM_SIZE = 1

# Nombre de processus RocketSim (CPU-bound). Monte si tu as plus de cores et
# que le GPU n'est pas saturé ; baisse si CPU à 100 %.
N_PROC = 10

# Render désactivé pour l'entraînement (perf catastrophique en multi-process).
# Pour visualiser : RENDER=True ET N_PROC=1 (un seul flux UDP).
RENDER = False  # False = entraînement rapide | True = visualiser (avec N_PROC=1)

# Vitesse de rendu quand RENDER=True (1 step = 8 ticks @120 Hz = 1/15 s réel).
#   0.067 -> x1 | 0.033 -> x2 | 0.017 -> x4 | 0.0 -> max
RENDER_REALTIME_DELAY = 1.0 / 15.0

# Architecture du réseau. Projet neuf (pas de contrainte de compat checkpoint) :
# [512,512,512] est sûr et rapide sur la plupart des GPU. Tu peux monter à
# [1024,1024,512] pour un plafond de skill plus haut si ton GPU suit (throughput
# plus lent → plus long pour atteindre les milliards de steps).
NET_SIZE = [512, 512, 512]

# Répertoires
_SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_FOLDER = os.path.join(_SCRIPT_DIR, "data", "checkpoints", PROJECT_NAME)
os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)

# Seuils de fin de phase (CIBLE TOTALE cumulative en steps, pas un delta).
PHASE_THRESHOLDS = {
    1: 100_000_000,      # 100M  → fin phase 1
    2: 500_000_000,      # 500M  → fin phase 2
    3: 1_500_000_000,    # 1.5B  → fin phase 3
    4: 5_000_000_000,    # 5B    → fin phase 4
    5: 10 ** 18,         # ∞     → phase 5 permanente
}
MIN_PHASE = 1
MAX_PHASE = 5

PHASE_NAMES = {
    1: "PHASE 1 — Bases (0 → 100M) : toucher la balle, ne pas oublier de sauter",
    2: "PHASE 2 — Domination SOL SSL (100M → 500M) : shots, dribbles, flicks, "
       "speedflip kickoff, saves, recovery, shadow defense, boost",
    3: "PHASE 3 — Sol maîtrisé + intro aérienne (500M → 1.5B)",
    4: "PHASE 4 — Mécas SSL (1.5B → 5B) : wall play, ceiling, air dribbles, "
       "flip resets, double taps (priorité SOL maintenue)",
    5: "PHASE 5 — SSL Grind (5B → ∞) : optimisation totale équilibrée",
}

# [1V1] selflessness (team_spirit) n'a AUCUN effet en 1v1 (pas de coéquipier).
# On le laisse à 0. opp_punish reste actif : c'est le "zero-sum" vs adversaire
# (style Necto/Nexto) qui croît au fil des phases.
TEAM_SPIRIT_BY_PHASE = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
OPP_PUNISH_BY_PHASE  = {1: 0.0, 2: 0.3, 3: 0.5, 4: 0.7, 5: 1.0}

# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRES PHASES
# ─────────────────────────────────────────────────────────────────────────────

_PHASE_STARTS = {1: 0, 2: 100_000_000, 3: 500_000_000,
                 4: 1_500_000_000, 5: 5_000_000_000}


def get_total_steps_from_checkpoints(folder: str) -> int:
    """Lit le nombre de steps depuis le dossier de checkpoint le plus avancé."""
    if not os.path.isdir(folder):
        return 0
    try:
        nums = [
            int(f) for f in os.listdir(folder)
            if f.isdigit() and os.path.isdir(os.path.join(folder, f))
        ]
        return max(nums) if nums else 0
    except OSError:
        return 0


def get_phase(total_steps: int) -> int:
    """Retourne la phase (1-5) selon les steps totaux."""
    phase = 1
    for p, start in sorted(_PHASE_STARTS.items()):
        if total_steps >= start:
            phase = p
    return phase


def find_latest_checkpoint(folder: str):
    """Retourne (chemin, steps) du checkpoint le plus avancé."""
    if not os.path.isdir(folder):
        return None, 0
    subdirs = []
    for name in os.listdir(folder):
        full = os.path.join(folder, name)
        if os.path.isdir(full) and name.isdigit():
            subdirs.append((int(name), full))
    if not subdirs:
        return None, 0
    subdirs.sort(key=lambda x: x[0])
    return subdirs[-1][1], subdirs[-1][0]


def _is_checkpoint_valid(folder_path: str) -> bool:
    """
    Valide qu'un checkpoint est utilisable (rlgym-ppo ne fait pas de save
    atomique → un Ctrl+C/OOM/crash en plein save peut tronquer le JSON et
    faire crasher le prochain learner.load() avec JSONDecodeError).

    Vérifie : JSON existe + non vide + parse en dict + clés requises + .pt OK.
    """
    required_keys = (
        "cumulative_timesteps",
        "policy_average_reward",
        "cumulative_model_updates",
        "reward_running_stats",
        "epoch",
    )
    bk_path = os.path.join(folder_path, "BOOK_KEEPING_VARS.json")
    try:
        if not os.path.isfile(bk_path) or os.path.getsize(bk_path) == 0:
            return False
        with open(bk_path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return False
        for k in required_keys:
            if k not in data:
                return False
    except (json.JSONDecodeError, OSError, ValueError):
        return False
    for name in ("PPO_POLICY.pt", "PPO_VALUE_NET.pt"):
        p = os.path.join(folder_path, name)
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            return False
    return True


def find_latest_valid_checkpoint(folder: str):
    """
    Comme find_latest_checkpoint, mais skip automatiquement les checkpoints
    corrompus et fallback sur le dernier VALIDE. N'efface RIEN automatiquement.
    """
    if not os.path.isdir(folder):
        return None, 0
    subdirs = []
    for name in os.listdir(folder):
        full = os.path.join(folder, name)
        if os.path.isdir(full) and name.isdigit():
            subdirs.append((int(name), full))
    if not subdirs:
        return None, 0
    subdirs.sort(key=lambda x: x[0], reverse=True)   # plus récent d'abord
    skipped = []
    for steps, path in subdirs:
        if _is_checkpoint_valid(path):
            if skipped:
                print(
                    f"[CHECKPOINT] Skipped {len(skipped)} corrompu(s) : "
                    f"{', '.join(f'{s:,}' for s, _ in skipped)}"
                )
                print(f"[CHECKPOINT] Fallback sur le dernier valide : {steps:,} steps")
            return path, steps
        skipped.append((steps, path))
    print(
        f"[CHECKPOINT] ATTENTION : {len(skipped)} checkpoint(s) trouvé(s) "
        f"mais TOUS corrompus. Démarrage from scratch."
    )
    return None, 0

# ─────────────────────────────────────────────────────────────────────────────
# RENDERER (RocketSimVis) — utilisé uniquement si RENDER=True
# ─────────────────────────────────────────────────────────────────────────────

from rlgym.api import Renderer
from rlgym.rocket_league.api import GameState, Car

BUTTON_NAMES = ("throttle", "steer", "pitch", "yaw", "roll", "jump", "boost", "handbrake")


class RocketSimVisRenderer(Renderer[GameState]):
    def __init__(self, udp_ip="127.0.0.1", udp_port=9273):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_ip  = udp_ip
        self.udp_port = udp_port

    @staticmethod
    def _phys(po):
        return {
            "pos":     po.position.tolist(),
            "forward": po.forward.tolist(),
            "up":      po.up.tolist(),
            "vel":     po.linear_velocity.tolist(),
            "ang_vel": po.angular_velocity.tolist(),
        }

    @staticmethod
    def _car(car: Car, controls=None):
        j = {
            "team_num":   int(car.team_num),
            "phys":       RocketSimVisRenderer._phys(car.physics),
            "boost_amount": car.boost_amount,
            "on_ground":  bool(car.on_ground),
            "has_flipped_or_double_jumped": bool(car.has_flipped or car.has_double_jumped),
            "is_demoed":  bool(car.is_demoed),
            "has_flip":   bool(car.can_flip),
        }
        if controls is not None:
            if isinstance(controls, np.ndarray):
                controls = {k: float(v) for k, v in zip(BUTTON_NAMES, controls)}
            j["controls"] = controls
        return j

    def render(self, state: GameState, shared_info: Dict[str, Any]) -> Any:
        controls = shared_info.get("controls", {})
        payload = {
            "ball_phys": self._phys(state.ball),
            "cars": [self._car(c, controls.get(aid)) for aid, c in state.cars.items()],
            "boost_pad_states": (state.boost_pad_timers <= 0).tolist(),
        }
        try:
            self.sock.sendto(json.dumps(payload).encode(), (self.udp_ip, self.udp_port))
        except OSError:
            pass

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass

# ─────────────────────────────────────────────────────────────────────────────
# STATE MUTATORS — situations d'entraînement
# ─────────────────────────────────────────────────────────────────────────────

from rlgym.api import StateMutator
from rlgym.rocket_league.common_values import (
    SIDE_WALL_X, BACK_WALL_Y, CEILING_Z,
    CAR_MAX_SPEED, BALL_MAX_SPEED, BACK_NET_Y
)
from rlgym.rocket_league.math import rand_vec3, rand_uvec3
from rlgym.rocket_league.state_mutators import (
    MutatorSequence, KickoffMutator, FixedTeamSizeMutator
)
from rlgym_tools.rocket_league.state_mutators.weighted_sample_mutator import WeightedSampleMutator
from rlgym_tools.rocket_league.reward_functions.aerial_distance_reward import RAMP_HEIGHT


def _rotation_from_forward(fw: np.ndarray) -> np.ndarray:
    """Matrice de rotation orthonormale stable depuis un vecteur forward."""
    fw = fw / (np.linalg.norm(fw) + 1e-8)
    up_hint = np.array([0., 0., 1.])
    if abs(np.dot(fw, up_hint)) > 0.95:
        up_hint = np.array([0., 1., 0.])
    rgt = np.cross(up_hint, fw)
    rgt = rgt / (np.linalg.norm(rgt) + 1e-8)
    up  = np.cross(fw, rgt)
    up  = up  / (np.linalg.norm(up)  + 1e-8)
    return np.stack([fw, rgt, up])


class RandomPhysicsMutator(StateMutator[GameState]):
    """Spawn aléatoire balle + voitures partout sur le terrain."""
    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        padding, goal_line_y, min_goal_dist = 100, 5120, 2000
        for i, po in enumerate([state.ball] + [c.physics for c in state.cars.values()]):
            while True:
                max_z = (CEILING_Z - padding) if i == 0 else (CEILING_Z / 6 - padding)
                pos = np.random.uniform(
                    [-SIDE_WALL_X + padding, -BACK_WALL_Y + padding, padding],
                    [ SIDE_WALL_X - padding,  BACK_WALL_Y - padding, max_z],
                )
                if i == 0 and abs(pos[1]) > goal_line_y - min_goal_dist:
                    continue
                if abs(pos[0]) + abs(pos[1]) >= 8064 - padding:
                    continue
                near_wall = (
                    abs(pos[0]) >= SIDE_WALL_X - RAMP_HEIGHT or
                    abs(pos[1]) >= BACK_WALL_Y - RAMP_HEIGHT or
                    abs(pos[0]) + abs(pos[1]) >= 8064 - RAMP_HEIGHT
                )
                near_surface = pos[2] <= RAMP_HEIGHT or pos[2] >= CEILING_Z - RAMP_HEIGHT
                if near_wall and near_surface:
                    continue
                break
            po.position         = pos
            po.linear_velocity  = rand_vec3(2300)
            po.angular_velocity = rand_vec3(5)
            if i > 0:
                po.rotation_mtx = _rotation_from_forward(rand_uvec3())


class GroundPlayMutator(StateMutator[GameState]):
    """
    [SOL] Situation de jeu au SOL : balle basse/au sol, voiture au sol derrière
    la balle avec boost et vélocité vers elle. Cœur de l'entraînement 1v1 :
    dribbles, conduite de balle, shots, 50/50.
    """
    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        padding = 200
        bx = np.random.uniform(-SIDE_WALL_X * 0.7, SIDE_WALL_X * 0.7)
        by = np.random.uniform(-BACK_WALL_Y * 0.6, BACK_WALL_Y * 0.6)
        bz = np.random.uniform(93.0, 300.0)  # balle au sol / petit rebond
        state.ball.position         = np.array([bx, by, bz])
        state.ball.linear_velocity  = np.array([
            np.random.uniform(-800, 800),
            np.random.uniform(-800, 800),
            np.random.uniform(0, 400),
        ])
        state.ball.angular_velocity = rand_vec3(2)

        for car in state.cars.values():
            # but adverse de cette voiture
            goal_y = -BACK_NET_Y if car.is_orange else BACK_NET_Y
            # derrière la balle (côté de son propre but)
            behind = -np.sign(goal_y) if goal_y != 0 else -1.0
            cx = np.clip(bx + np.random.uniform(-700, 700),
                         -SIDE_WALL_X + padding, SIDE_WALL_X - padding)
            cy = np.clip(by + behind * np.random.uniform(600, 1600),
                         -BACK_WALL_Y + padding, BACK_WALL_Y - padding)
            car.physics.position = np.array([cx, cy, 17.0])
            dir_to_ball = np.array([bx - cx, by - cy, 0.0])
            dn = np.linalg.norm(dir_to_ball)
            if dn > 1e-6:
                dir_to_ball = dir_to_ball / dn
            car.physics.linear_velocity  = dir_to_ball * np.random.uniform(0, 1200)
            car.physics.angular_velocity = np.zeros(3)
            car.physics.rotation_mtx     = _rotation_from_forward(dir_to_ball)
            car.boost_amount             = float(np.random.uniform(20, 100))


class ShadowDefenseMutator(StateMutator[GameState]):
    """
    [SOL/DÉFENSE] L'adversaire a la balle dans la moitié de TikshiV1 → entraîne
    la shadow defense, le positionnement entre balle et but, le challenge.
    """
    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        # Balle dans une moitié aléatoire, plutôt basse
        bx = np.random.uniform(-SIDE_WALL_X * 0.6, SIDE_WALL_X * 0.6)
        side = np.random.choice([-1.0, 1.0])
        by = side * np.random.uniform(1500, 4200)
        bz = np.random.uniform(93.0, 500.0)
        state.ball.position         = np.array([bx, by, bz])
        state.ball.linear_velocity  = np.array([
            np.random.uniform(-600, 600),
            side * np.random.uniform(200, 1400),   # balle qui avance vers un but
            np.random.uniform(0, 300),
        ])
        state.ball.angular_velocity = rand_vec3(2)

        padding = 200
        for car in state.cars.values():
            own_goal_y = BACK_NET_Y if car.is_orange else -BACK_NET_Y
            # défenseur : entre la balle et son propre but
            cx = np.clip(bx * 0.5 + np.random.uniform(-700, 700),
                         -SIDE_WALL_X + padding, SIDE_WALL_X - padding)
            cy = np.clip((by + own_goal_y) * 0.5 + np.random.uniform(-500, 500),
                         -BACK_WALL_Y + padding, BACK_WALL_Y - padding)
            car.physics.position = np.array([cx, cy, 17.0])
            dir_to_ball = np.array([bx - cx, by - cy, 0.0])
            dn = np.linalg.norm(dir_to_ball)
            if dn > 1e-6:
                dir_to_ball = dir_to_ball / dn
            car.physics.linear_velocity  = dir_to_ball * np.random.uniform(0, 900)
            car.physics.angular_velocity = np.zeros(3)
            car.physics.rotation_mtx     = _rotation_from_forward(dir_to_ball)
            car.boost_amount             = float(np.random.uniform(15, 80))


class WallPlayMutator(StateMutator[GameState]):
    """
    [MUR] Balle sur/le long d'un mur latéral. Voiture au sol avec gros boost →
    apprend le wall play ET le choix « partir en aérienne si beaucoup de boost »
    (exactement le cas demandé : sur un mur + beaucoup de boost → aérienne).
    """
    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        wall_side = np.random.choice([-1.0, 1.0])
        bx = wall_side * np.random.uniform(SIDE_WALL_X - 600, SIDE_WALL_X - 120)
        by = np.random.uniform(-BACK_WALL_Y * 0.5, BACK_WALL_Y * 0.5)
        bz = np.random.uniform(300, 1500)
        state.ball.position         = np.array([bx, by, bz])
        state.ball.linear_velocity  = np.array([
            -wall_side * np.random.uniform(0, 400),
            np.random.uniform(-600, 600),
            np.random.uniform(-200, 400),
        ])
        state.ball.angular_velocity = rand_vec3(2)

        padding = 200
        for car in state.cars.values():
            cx = np.clip(bx - wall_side * np.random.uniform(300, 1100),
                         -SIDE_WALL_X + padding, SIDE_WALL_X - padding)
            cy = np.clip(by + np.random.uniform(-900, 900),
                         -BACK_WALL_Y + padding, BACK_WALL_Y - padding)
            car.physics.position = np.array([cx, cy, 17.0])
            dir_to_ball = np.array([bx - cx, by - cy, 0.0])
            dn = np.linalg.norm(dir_to_ball)
            if dn > 1e-6:
                dir_to_ball = dir_to_ball / dn
            car.physics.linear_velocity  = dir_to_ball * np.random.uniform(0, 1000)
            car.physics.angular_velocity = np.zeros(3)
            car.physics.rotation_mtx     = _rotation_from_forward(dir_to_ball)
            # gros boost → l'option aérienne devient pertinente
            car.boost_amount             = float(np.random.uniform(60, 100))


class AerialSetupMutator(StateMutator[GameState]):
    """
    [AÉRIEN] Balle haute, voiture AU SOL face à la balle, boost plein, vélocité
    initiale vers elle. Apprend à décoller proprement pour des aériennes.
    """
    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        padding = 150
        bx = np.random.uniform(-SIDE_WALL_X * 0.55, SIDE_WALL_X * 0.55)
        by = np.random.uniform(-BACK_WALL_Y * 0.45, BACK_WALL_Y * 0.45)
        bz = np.random.uniform(600, 1700)

        state.ball.position         = np.array([bx, by, bz])
        state.ball.linear_velocity  = rand_vec3(np.random.uniform(400, 1600))
        state.ball.angular_velocity = rand_vec3(2)

        for car in state.cars.values():
            while True:
                cx = bx + np.random.uniform(-1000, 1000)
                cy = by + np.random.uniform(-1000, 1000)
                cx = np.clip(cx, -SIDE_WALL_X + padding, SIDE_WALL_X - padding)
                cy = np.clip(cy, -BACK_WALL_Y + padding, BACK_WALL_Y - padding)
                if abs(cx) + abs(cy) < 8064 - padding:
                    break
            car.physics.position         = np.array([cx, cy, 17.0])
            dir_to_ball = np.array([bx - cx, by - cy, 0.0])
            dn          = np.linalg.norm(dir_to_ball)
            if dn > 1e-6:
                dir_to_ball = dir_to_ball / dn
            launch_spd = np.random.uniform(600, 1400)
            car.physics.linear_velocity  = dir_to_ball * launch_spd
            car.physics.angular_velocity = np.zeros(3)
            car.physics.rotation_mtx     = _rotation_from_forward(
                np.array([bx - cx, by - cy, bz - 17.0])
            )
            car.boost_amount = 100.0


class AerialAirCarMutator(StateMutator[GameState]):
    """
    [AÉRIEN AVANCÉ] Voitures déjà EN L'AIR autour de la balle (haute), vélocité
    initiale + boost. Apprend pinches, double touches, redirects, flip resets,
    air dribbles. Réservé phases 4-5, en MINORITÉ (l'aérien n'est pas la spé).
    """
    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        bx = np.random.uniform(-SIDE_WALL_X * 0.5, SIDE_WALL_X * 0.5)
        by = np.random.uniform(-BACK_WALL_Y * 0.4, BACK_WALL_Y * 0.4)
        bz = np.random.uniform(900, 1900)
        state.ball.position         = np.array([bx, by, bz])
        state.ball.linear_velocity  = rand_vec3(np.random.uniform(300, 1400))
        state.ball.angular_velocity = rand_vec3(3)

        for car in state.cars.values():
            offset = np.random.uniform([-700, -700, -500], [700, 700, 200])
            pos = np.array([bx, by, bz]) + offset
            pos[2] = np.clip(pos[2], 200, CEILING_Z - 200)
            pos[0] = np.clip(pos[0], -SIDE_WALL_X + 200, SIDE_WALL_X - 200)
            pos[1] = np.clip(pos[1], -BACK_WALL_Y + 200, BACK_WALL_Y - 200)

            car.physics.position = pos
            dir_to_ball = np.array([bx, by, bz]) - pos
            dn = np.linalg.norm(dir_to_ball)
            if dn > 1e-6:
                dir_to_ball = dir_to_ball / dn
            spd = np.random.uniform(800, 1600)
            car.physics.linear_velocity  = dir_to_ball * spd
            car.physics.angular_velocity = rand_vec3(3)
            car.physics.rotation_mtx     = _rotation_from_forward(dir_to_ball)
            car.boost_amount             = float(np.random.uniform(50, 100))


def build_state_mutator(phase: int):
    """
    [1V1] FixedTeamSizeMutator(blue=1, orange=1).

    Distribution SOL-FIRST : à toutes les phases, la majorité des situations
    sont au sol. L'aérien est présent mais minoritaire (jamais la spécialité).
    """
    base = FixedTeamSizeMutator(blue_size=TEAM_SIZE, orange_size=TEAM_SIZE)

    if phase == 1:
        # Bases : surtout kickoff + sol, un peu de chaos.
        inner = WeightedSampleMutator.from_zipped(
            (KickoffMutator(),       0.45),
            (GroundPlayMutator(),    0.35),
            (RandomPhysicsMutator(), 0.20),
        )
    elif phase == 2:
        # Domination sol : kickoff (speedflip), jeu au sol, shadow defense.
        inner = WeightedSampleMutator.from_zipped(
            (KickoffMutator(),        0.30),
            (GroundPlayMutator(),     0.40),
            (ShadowDefenseMutator(),  0.18),
            (RandomPhysicsMutator(),  0.12),
        )
    elif phase == 3:
        # Sol maîtrisé + intro aérienne (sol reste majoritaire ~70 %).
        inner = WeightedSampleMutator.from_zipped(
            (KickoffMutator(),        0.22),
            (GroundPlayMutator(),     0.33),
            (ShadowDefenseMutator(),  0.15),
            (WallPlayMutator(),       0.10),
            (AerialSetupMutator(),    0.15),
            (RandomPhysicsMutator(),  0.05),
        )
    elif phase == 4:
        # Mécas avancées, mais SOL toujours majoritaire (~60 %).
        inner = WeightedSampleMutator.from_zipped(
            (KickoffMutator(),        0.18),
            (GroundPlayMutator(),     0.27),
            (ShadowDefenseMutator(),  0.15),
            (WallPlayMutator(),       0.15),
            (AerialSetupMutator(),    0.15),
            (AerialAirCarMutator(),   0.10),
        )
    else:  # phase 5 — SSL grind équilibré (sol ~60 %)
        inner = WeightedSampleMutator.from_zipped(
            (KickoffMutator(),        0.18),
            (GroundPlayMutator(),     0.27),
            (ShadowDefenseMutator(),  0.15),
            (WallPlayMutator(),       0.15),
            (AerialSetupMutator(),    0.13),
            (AerialAirCarMutator(),   0.12),
        )

    return MutatorSequence(base, inner)

# ─────────────────────────────────────────────────────────────────────────────
# REWARD FUNCTIONS — SOL (toujours actives, poids forts)
# ─────────────────────────────────────────────────────────────────────────────

from rlgym.api import RewardFunction
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward


class FaceBallReward(RewardFunction):
    """Le bot regarde la balle — signal directionnel léger."""
    def reset(self, agents, initial_state, shared_info): pass
    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        for agent in agents:
            car  = state.cars[agent]
            diff = state.ball.position - car.physics.position
            dist = np.linalg.norm(diff)
            rewards[agent] = float(np.dot(car.physics.forward, diff / dist)) if dist > 0 else 0.0
        return rewards


class SpeedTowardBallReward(RewardFunction):
    """Vitesse projetée vers la balle, normalisée par CAR_MAX_SPEED (>= 0)."""
    def reset(self, agents, initial_state, shared_info): pass
    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        for agent in agents:
            car  = state.cars[agent]
            diff = state.ball.position - car.physics.position
            dist = np.linalg.norm(diff)
            if dist < 1e-6:
                rewards[agent] = 0.0
                continue
            spd = np.dot(car.physics.linear_velocity, diff / dist)
            rewards[agent] = max(spd / CAR_MAX_SPEED, 0.0)
        return rewards


class VelocityBallToGoalReward(RewardFunction):
    """Vitesse de la balle vers le but adverse."""
    def reset(self, agents, initial_state, shared_info): pass
    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        for agent in agents:
            car    = state.cars[agent]
            goal_y = -BACK_NET_Y if car.is_orange else BACK_NET_Y
            diff   = np.array([0., goal_y, 0.]) - state.ball.position
            dist   = np.linalg.norm(diff)
            if dist < 1e-6:
                rewards[agent] = 0.0
                continue
            vel = np.dot(state.ball.linear_velocity, diff / dist)
            rewards[agent] = max(vel / BALL_MAX_SPEED, 0.0)
        return rewards


class AdvancedTouchReward(RewardFunction):
    """Touche pondérée par l'accélération donnée à la balle (force du contact)."""
    def __init__(self, touch_weight=0.0, accel_weight=1.0):
        self.touch_weight  = touch_weight
        self.accel_weight  = accel_weight
        self._prev_ball_vel = None

    def reset(self, agents, initial_state, shared_info):
        self._prev_ball_vel = initial_state.ball.linear_velocity.copy()

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {a: 0.0 for a in agents}
        if self._prev_ball_vel is None:
            self._prev_ball_vel = state.ball.linear_velocity.copy()
        accel = np.linalg.norm(state.ball.linear_velocity - self._prev_ball_vel) / BALL_MAX_SPEED
        for agent in agents:
            if state.cars[agent].ball_touches > 0:
                rewards[agent] = self.touch_weight + self.accel_weight * accel
        self._prev_ball_vel = state.ball.linear_velocity.copy()
        return rewards


class SaveBoostReward(RewardFunction):
    """sqrt(boost/100) → 0-1. Le boost vaut plus cher quand il est bas (sqrt)."""
    def reset(self, agents, initial_state, shared_info): pass
    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        return {
            a: math.sqrt(max(state.cars[a].boost_amount / 100.0, 0.0))
            for a in agents
        }


class BoostPickupReward(RewardFunction):
    """Gros pad (gain > 30) → 1.0  |  Petit pad → 0.6"""
    BIG_THRESHOLD  = 30.0
    BIG_REWARD     = 1.0
    SMALL_REWARD   = 0.6
    NOISE_FLOOR    = 5.0

    def __init__(self):
        self._prev = {}

    def reset(self, agents, initial_state, shared_info):
        self._prev = {a: initial_state.cars[a].boost_amount for a in agents}

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {a: 0.0 for a in agents}
        for agent in agents:
            curr  = state.cars[agent].boost_amount
            prev  = self._prev.get(agent, curr)
            gain  = curr - prev
            if gain > self.NOISE_FLOOR:
                rewards[agent] = self.BIG_REWARD if gain > self.BIG_THRESHOLD else self.SMALL_REWARD
            self._prev[agent] = curr
        return rewards


class InAirReward(RewardFunction):
    """Signal léger anti « forget how to jump » (RLGym-PPO-Guide)."""
    def reset(self, agents, initial_state, shared_info): pass
    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        return {a: float(not state.cars[a].on_ground) for a in agents}


class SaveReward(RewardFunction):
    """Déviation d'un tir allant vers ses propres buts."""
    def __init__(self, min_ball_speed=500.0):
        self.min_ball_speed = min_ball_speed
        self._prev_vel = None

    def reset(self, agents, initial_state, shared_info):
        self._prev_vel = initial_state.ball.linear_velocity.copy()

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {a: 0.0 for a in agents}
        if self._prev_vel is None:
            self._prev_vel = state.ball.linear_velocity.copy()
        prev  = self._prev_vel
        curr  = state.ball.linear_velocity
        speed = np.linalg.norm(prev)
        if speed > self.min_ball_speed:
            for agent in agents:
                if state.cars[agent].ball_touches <= 0:
                    continue
                car        = state.cars[agent]
                own_goal_y = BACK_NET_Y if car.is_orange else -BACK_NET_Y
                going_own  = np.sign(prev[1]) == np.sign(own_goal_y)
                reversed_  = np.sign(curr[1]) != np.sign(prev[1]) or abs(curr[1]) < abs(prev[1]) * 0.5
                if going_own and reversed_:
                    rewards[agent] = 1.5
        self._prev_vel = curr.copy()
        return rewards


class FlickReward(RewardFunction):
    """Touche puissante avec balle entre 150-400 UU au-dessus de la voiture."""
    def __init__(self, min_h=150.0, max_h=400.0):
        self.min_h = min_h
        self.max_h = max_h
        self._prev_ball_vel = None

    def reset(self, agents, initial_state, shared_info):
        self._prev_ball_vel = initial_state.ball.linear_velocity.copy()

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {a: 0.0 for a in agents}
        if self._prev_ball_vel is None:
            self._prev_ball_vel = state.ball.linear_velocity.copy()
        ball  = state.ball
        accel = np.linalg.norm(ball.linear_velocity - self._prev_ball_vel) / BALL_MAX_SPEED
        for agent in agents:
            car = state.cars[agent]
            if car.ball_touches <= 0:
                continue
            rel_h = ball.position[2] - car.physics.position[2]
            if self.min_h < rel_h < self.max_h:
                goal_y   = -BACK_NET_Y if car.is_orange else BACK_NET_Y
                dir_goal = np.array([0., goal_y, 0.]) - ball.position
                dn       = np.linalg.norm(dir_goal)
                shot_q   = max(np.dot(ball.linear_velocity / (BALL_MAX_SPEED + 1e-8),
                                      dir_goal / (dn + 1e-8)), 0.0)
                rewards[agent] = accel * (1.0 + shot_q)
        self._prev_ball_vel = ball.linear_velocity.copy()
        return rewards


class DribbleReward(RewardFunction):
    """Balle sur le toit de la voiture et proche — encourage le dribble (sol)."""
    def __init__(self, min_h=80.0, max_h=300.0, max_xy=250.0):
        self.min_h  = min_h
        self.max_h  = max_h
        self.max_xy = max_xy

    def reset(self, agents, initial_state, shared_info): pass
    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        for agent in agents:
            car     = state.cars[agent]
            rel_h   = state.ball.position[2] - car.physics.position[2]
            xy_dist = np.linalg.norm(state.ball.position[:2] - car.physics.position[:2])
            # [SOL-FIRST] dribble compte uniquement si la voiture est au sol :
            # on récompense porter la balle au sol, pas en l'air.
            if car.on_ground and self.min_h < rel_h < self.max_h and xy_dist < self.max_xy:
                h_score = 1.0 - abs(rel_h - 150) / (self.max_h - self.min_h)
                d_score = 1.0 - xy_dist / self.max_xy
                rewards[agent] = max(0.0, 0.5 * h_score + 0.5 * d_score)
            else:
                rewards[agent] = 0.0
        return rewards


class SpeedflipKickoffReward(RewardFunction):
    """
    [KICKOFF SSL] Récompense un kickoff RAPIDE et agressif (speedflip).
    Actif tant que la balle est au centre (kickoff en cours) :
      - vitesse de la voiture (le speedflip atteint la supersonique vite),
      - bonus supersonique,
      - bonus si un dodge/flip a été utilisé (cœur du speedflip),
      - gros bonus sur la première touche (gagner le kickoff).
    Pénalité d'air très faible : les flip-kickoffs SSL sont efficaces.
    """
    BALL_CENTER_R = 300.0

    def __init__(self, touch_bonus=1.0, speed_coef=0.5, flip_bonus=0.3,
                 supersonic_bonus=0.3, air_penalty=-0.05):
        self.touch_bonus      = touch_bonus
        self.speed_coef       = speed_coef
        self.flip_bonus       = flip_bonus
        self.supersonic_bonus = supersonic_bonus
        self.air_penalty      = air_penalty
        self._done            = False
        self._flipped         = {}

    def reset(self, agents, initial_state, shared_info):
        self._done    = False
        self._flipped = {a: False for a in agents}

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {a: 0.0 for a in agents}
        if self._done:
            return rewards
        ball    = state.ball
        ball_xy = np.linalg.norm(ball.position[:2])
        if ball_xy > self.BALL_CENTER_R:
            self._done = True
            return rewards
        for agent in agents:
            car = state.cars[agent]
            if car.has_flipped or car.is_flipping:
                self._flipped[agent] = True
            if car.ball_touches > 0:
                r = self.touch_bonus
                if self._flipped.get(agent, False):
                    r += self.flip_bonus            # a fait un (speed)flip → bonus
                rewards[agent] = r
                self._done     = True
                return rewards
            # approche : récompense la vitesse vers la balle
            diff = ball.position - car.physics.position
            dist = np.linalg.norm(diff)
            spd_to_ball = np.dot(car.physics.linear_velocity, diff / dist) if dist > 1e-6 else 0.0
            r = float(np.clip(spd_to_ball / CAR_MAX_SPEED * self.speed_coef,
                              0.0, self.speed_coef))
            if car.is_supersonic:
                r += self.supersonic_bonus
            if not car.on_ground and not car.is_flipping:
                r += self.air_penalty
            rewards[agent] = r
        return rewards


class RecoveryReward(RewardFunction):
    """
    [RECOVERY SSL] Quand la voiture est en l'air et NE touche pas la balle,
    récompense l'orientation pour un atterrissage propre (wheels-down) et
    l'alignement du nez sur la vélocité horizontale. Encourage les recovery
    rapides (sortie d'aérienne / après dodge) → repartir vite au sol.
    """
    def reset(self, agents, initial_state, shared_info): pass
    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {a: 0.0 for a in agents}
        for agent in agents:
            car = state.cars[agent]
            if car.on_ground:
                continue
            up_align = float(car.physics.up[2])               # 1 = roues vers le bas
            vel = car.physics.linear_velocity
            vxy = np.array([vel[0], vel[1], 0.0])
            n   = np.linalg.norm(vxy)
            if n > 200.0:
                fwd = car.physics.forward
                nose_align = float(np.dot(fwd / (np.linalg.norm(fwd) + 1e-8), vxy / n))
            else:
                nose_align = 0.0
            rewards[agent] = max(0.0, 0.5 * up_align + 0.5 * nose_align)
        return rewards

# ─────────────────────────────────────────────────────────────────────────────
# REWARD FUNCTIONS — POSITIONNEMENT / SHADOW DEFENSE / SHAPING
# ─────────────────────────────────────────────────────────────────────────────


class DefendGoalReward(RewardFunction):
    """
    Récompense le positionnement entre la balle et son propre but quand la
    balle est dans sa moitié (shadow defense 1v1).
    """
    def reset(self, agents, initial_state, shared_info): pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        ball = state.ball
        for agent in agents:
            car        = state.cars[agent]
            own_goal_y = BACK_NET_Y if car.is_orange else -BACK_NET_Y
            ball_in_own_half = np.sign(ball.position[1]) == np.sign(own_goal_y)
            if ball_in_own_half:
                goal_pos  = np.array([0., own_goal_y, 0.])
                to_goal   = goal_pos - ball.position
                to_goal_n = to_goal / (np.linalg.norm(to_goal) + 1e-6)
                to_car    = car.physics.position - ball.position
                to_car_n  = to_car  / (np.linalg.norm(to_car)  + 1e-6)
                alignment = np.dot(to_car_n, to_goal_n)
                rewards[agent] = max(alignment, 0.0) * 0.5
            else:
                rewards[agent] = 0.0
        return rewards


class GoalDistancePotentialReward(RewardFunction):
    """
    Potential-based shaping (Ng et al. 1999) — préserve la politique optimale.
    Φ(s) = 0.5 * (exp(-d_to_orange/CAR_MAX_SPEED) - exp(-d_to_blue/CAR_MAX_SPEED))
    Reward = Φ(s') - Φ(s). Pousse la balle vers le but adverse sans bruit.
    """
    BLUE_GOAL   = np.array([0., -BACK_NET_Y, 100.])
    ORANGE_GOAL = np.array([0.,  BACK_NET_Y, 100.])

    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma
        self._prev_phi = None

    @staticmethod
    def _phi(state) -> float:
        ball = state.ball.position
        d_orange = np.linalg.norm(ball - GoalDistancePotentialReward.ORANGE_GOAL)
        d_blue   = np.linalg.norm(ball - GoalDistancePotentialReward.BLUE_GOAL)
        return 0.5 * (math.exp(-d_orange / CAR_MAX_SPEED)
                      - math.exp(-d_blue   / CAR_MAX_SPEED))

    def reset(self, agents, initial_state, shared_info):
        self._prev_phi = self._phi(initial_state)

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        phi = self._phi(state)
        delta = self.gamma * phi - (self._prev_phi if self._prev_phi is not None else phi)
        self._prev_phi = phi
        return {
            a: float(-delta if state.cars[a].is_orange else delta)
            for a in agents
        }


class BehindBallReward(RewardFunction):
    """Récompense la position DERRIÈRE la balle (entre balle et propre but)."""
    def reset(self, agents, initial_state, shared_info): pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        ball_y = state.ball.position[1]
        for agent in agents:
            car = state.cars[agent]
            own_goal_y = BACK_NET_Y if car.is_orange else -BACK_NET_Y
            car_y = car.physics.position[1]
            if own_goal_y < 0:
                rewards[agent] = 1.0 if car_y < ball_y else 0.0
            else:
                rewards[agent] = 1.0 if car_y > ball_y else 0.0
        return rewards


class ChallengeReward(RewardFunction):
    """
    [50/50 SSL] Récompense d'aller au contact quand l'adversaire est lui aussi
    proche de la balle (challenge game). Le bot apprend à gagner les 50/50
    plutôt que de les fuir. Récompense = touche gagnée dans une situation
    contestée (adversaire à < contest_radius de la balle).
    """
    def __init__(self, contest_radius: float = 800.0):
        self.contest_radius = contest_radius

    def reset(self, agents, initial_state, shared_info): pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {a: 0.0 for a in agents}
        ball_pos = state.ball.position
        for agent in agents:
            car = state.cars[agent]
            if car.ball_touches <= 0:
                continue
            # un adversaire est-il proche de la balle ?
            contested = False
            for oid, other in state.cars.items():
                if oid == agent or other.team_num == car.team_num:
                    continue
                if np.linalg.norm(ball_pos - other.physics.position) < self.contest_radius:
                    contested = True
                    break
            if contested:
                rewards[agent] = 1.0
        return rewards

# ─────────────────────────────────────────────────────────────────────────────
# REWARDS AÉRIENS — excellents mais SECONDAIRES (plafonnés + conditionnés)
# ─────────────────────────────────────────────────────────────────────────────
#
# Règle d'or (demande utilisateur) : « les mécaniques aériennes doivent être
# excellentes mais ne jamais être prioritaires par rapport à une solution plus
# efficace au sol ». Concrètement :
#   - ces rewards ne se déclenchent QUE balle haute / voiture en l'air ;
#   - aucune pénalité pour rester au sol (pas de TouchGrassPenalty) ;
#   - leurs POIDS sont plafonnés sous les rewards de sol dans build_reward_fn.
# ─────────────────────────────────────────────────────────────────────────────


class AerialTouchReward(RewardFunction):
    """
    Touche aérienne de QUALITÉ : balle haute + voiture en l'air. Pondérée par
    min(air_time_frac, height_frac) (RLGym-PPO-Guide) → favorise les vraies
    aériennes (longues, hautes) plutôt que les pop de mur plats.
    """
    MAX_AIR_TIME = 1.75  # s — estimation d'un temps d'aérienne raisonnable

    def __init__(self, min_height=400.0):
        self.min_height     = min_height
        self._prev_ball_vel = None

    def reset(self, agents, initial_state, shared_info):
        self._prev_ball_vel = initial_state.ball.linear_velocity.copy()

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {a: 0.0 for a in agents}
        if self._prev_ball_vel is None:
            self._prev_ball_vel = state.ball.linear_velocity.copy()
        ball  = state.ball
        accel = np.linalg.norm(ball.linear_velocity - self._prev_ball_vel) / BALL_MAX_SPEED
        for agent in agents:
            car = state.cars[agent]
            if (car.ball_touches > 0
                    and not car.on_ground
                    and ball.position[2] > self.min_height):
                air_time_frac = min(car.air_time_since_jump, self.MAX_AIR_TIME) / self.MAX_AIR_TIME
                height_frac   = min(ball.position[2] / CEILING_Z, 1.0)
                quality       = min(air_time_frac, height_frac)
                rewards[agent] = accel * (1.0 + quality)
        self._prev_ball_vel = ball.linear_velocity.copy()
        return rewards


class AerialNavigationReward(RewardFunction):
    """En l'air : récompense face + vitesse vers la balle (guide le vol)."""
    def reset(self, agents, initial_state, shared_info): pass
    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            # [SOL-FIRST] uniquement en l'air ET balle assez haute pour justifier
            # une aérienne (sinon 0 → on ne pousse pas l'aérien sur balle basse).
            if car.on_ground or state.ball.position[2] < 300.0:
                rewards[agent] = 0.0
                continue
            diff = state.ball.position - car.physics.position
            dist = np.linalg.norm(diff)
            if dist < 1e-6:
                rewards[agent] = 0.0
                continue
            d     = diff / dist
            face  = float(np.dot(car.physics.forward, d))
            speed = max(np.dot(car.physics.linear_velocity, d) / CAR_MAX_SPEED, 0.0)
            rewards[agent] = 0.5 * face + 0.5 * speed
        return rewards


class CatchReward(RewardFunction):
    """La balle atterrit DOUCEMENT sur le capot (setup de dribble/flick)."""
    def __init__(self, max_rel_v_z=400.0, min_h=80.0, max_h=220.0, max_xy=160.0):
        self.max_rel_v_z = max_rel_v_z
        self.min_h       = min_h
        self.max_h       = max_h
        self.max_xy      = max_xy

    def reset(self, agents, initial_state, shared_info): pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {a: 0.0 for a in agents}
        for agent in agents:
            car = state.cars[agent]
            rel = state.ball.position - car.physics.position
            xy  = float(np.linalg.norm(rel[:2]))
            rel_h = float(rel[2])
            if self.min_h < rel_h < self.max_h and xy < self.max_xy:
                rel_v_z = abs(float(state.ball.linear_velocity[2]
                                    - car.physics.linear_velocity[2]))
                if rel_v_z < self.max_rel_v_z:
                    softness  = 1.0 - rel_v_z / self.max_rel_v_z
                    centering = 1.0 - xy / self.max_xy
                    rewards[agent] = softness * centering
        return rewards


class PerfectFlickReward(RewardFunction):
    """
    Flick « parfait » : balle sur le capot la frame d'avant, dodge maintenant,
    et la balle repart vite vers le but adverse. Plus exigeant que FlickReward.
    """
    def __init__(self, min_h=80.0, max_h=220.0, max_xy=200.0,
                 min_accel=400.0, min_dir_goal=0.3):
        self.min_h        = min_h
        self.max_h        = max_h
        self.max_xy       = max_xy
        self.min_accel    = min_accel
        self.min_dir_goal = min_dir_goal
        self._prev        = {}

    def reset(self, agents, initial_state, shared_info):
        v = initial_state.ball.linear_velocity.copy()
        self._prev = {a: {"on_hood": False, "ball_v": v.copy()} for a in agents}

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {a: 0.0 for a in agents}
        ball    = state.ball
        for agent in agents:
            car  = state.cars[agent]
            prev = self._prev.get(
                agent, {"on_hood": False, "ball_v": ball.linear_velocity.copy()}
            )
            rel  = ball.position - car.physics.position
            xy   = float(np.linalg.norm(rel[:2]))
            on_hood_now = (self.min_h < rel[2] < self.max_h) and (xy < self.max_xy)

            if prev["on_hood"] and car.is_flipping and car.ball_touches > 0:
                accel = float(np.linalg.norm(ball.linear_velocity - prev["ball_v"]))
                if accel > self.min_accel:
                    goal_y  = -BACK_NET_Y if car.is_orange else BACK_NET_Y
                    to_goal = np.array([0., goal_y, 0.]) - ball.position
                    dn      = float(np.linalg.norm(to_goal))
                    dir_q   = max(float(np.dot(
                        ball.linear_velocity / (BALL_MAX_SPEED + 1e-8),
                        to_goal / (dn + 1e-8)
                    )), 0.0)
                    if dir_q > self.min_dir_goal:
                        rewards[agent] = (accel / BALL_MAX_SPEED) * (1.0 + 2.0 * dir_q)

            self._prev[agent] = {
                "on_hood": on_hood_now,
                "ball_v":  ball.linear_velocity.copy(),
            }
        return rewards


class AirDribbleReward(RewardFunction):
    """
    Air dribble : touches CONSÉCUTIVES en l'air, délai court, vitesse cohérente.
    Streak capé à 5 pour ne pas exploser.
    """
    def __init__(self, min_ball_z=400.0, max_ticks_between=24,
                 min_velocity=800.0, max_streak=5.0):
        self.min_ball_z        = min_ball_z
        self.max_ticks_between = max_ticks_between
        self.min_velocity      = min_velocity
        self.max_streak        = max_streak
        self._streak           = {}
        self._ticks_since      = {}

    def reset(self, agents, initial_state, shared_info):
        self._streak      = {a: 0 for a in agents}
        self._ticks_since = {a: 1_000_000 for a in agents}

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards    = {a: 0.0 for a in agents}
        ball_z     = float(state.ball.position[2])
        ball_air   = ball_z > self.min_ball_z
        ball_speed = float(np.linalg.norm(state.ball.linear_velocity))
        for agent in agents:
            car = state.cars[agent]
            self._ticks_since[agent] = self._ticks_since.get(agent, 1_000_000) + 1
            touched_in_air = (
                car.ball_touches > 0
                and not car.on_ground
                and ball_air
                and ball_speed > self.min_velocity
            )
            if touched_in_air:
                if self._ticks_since[agent] <= self.max_ticks_between:
                    self._streak[agent] = min(
                        self._streak.get(agent, 0) + 1,
                        int(self.max_streak),
                    )
                    rewards[agent] = float(self._streak[agent])
                else:
                    self._streak[agent] = 1
                self._ticks_since[agent] = 0
            elif car.on_ground or not ball_air:
                self._streak[agent] = 0
        return rewards


class CeilingShotReward(RewardFunction):
    """
    Ceiling shot : la voiture passe haut (z >= 1700) puis redescend ET touche
    la balle avec une grosse accélération. Proportionnel à la puissance du tir.
    """
    def __init__(self, ceiling_z_threshold=1700.0, decay_frames=60, min_accel=800.0):
        self.ceiling_z_threshold = ceiling_z_threshold
        self.decay_frames        = decay_frames
        self.min_accel           = min_accel
        self._frames_since_top   = {}
        self._prev_ball_v        = None

    def reset(self, agents, initial_state, shared_info):
        self._frames_since_top = {a: 1_000_000 for a in agents}
        self._prev_ball_v      = initial_state.ball.linear_velocity.copy()

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {a: 0.0 for a in agents}
        if self._prev_ball_v is None:
            self._prev_ball_v = state.ball.linear_velocity.copy()
        ball_accel = float(np.linalg.norm(state.ball.linear_velocity - self._prev_ball_v))
        for agent in agents:
            car = state.cars[agent]
            self._frames_since_top[agent] = self._frames_since_top.get(agent, 1_000_000) + 1
            if float(car.physics.position[2]) >= self.ceiling_z_threshold:
                self._frames_since_top[agent] = 0
            if (car.ball_touches > 0
                    and self._frames_since_top[agent] <= self.decay_frames
                    and ball_accel > self.min_accel):
                rewards[agent] = ball_accel / BALL_MAX_SPEED
        self._prev_ball_v = state.ball.linear_velocity.copy()
        return rewards

# ─────────────────────────────────────────────────────────────────────────────
# BUILDER DE REWARDS PAR PHASE — SOL-FIRST, aérien plafonné
# ─────────────────────────────────────────────────────────────────────────────
#
# Philosophie (RLGym-PPO-Guide, Lucy-SKG, Necto) :
#   1) Pas de GoalReward écrasant (sinon noise, exploration réduite).
#   2) Touch pondéré par la force (AdvancedTouch).
#   3) Zero-sum (opp_punish) pour ce que l'adversaire doit empêcher : goals,
#      demos, flip resets, powershots.
#   4) Les rewards de SOL dominent toujours les rewards aériens (poids).
# ─────────────────────────────────────────────────────────────────────────────

from rlgym_tools.rocket_league.reward_functions.distribute_rewards_wrapper import (
    DistributeRewardsWrapper
)
from rlgym_tools.rocket_league.reward_functions.demo_reward import DemoReward
from rlgym_tools.rocket_league.reward_functions.flip_reset_reward import FlipResetReward
from rlgym_tools.rocket_league.reward_functions.aerial_distance_reward import (
    AerialDistanceReward
)
from rlgym_tools.rocket_league.reward_functions.wavedash_reward import WavedashReward


def _zero_sum(reward_fn, opp_punish: float):
    """
    Enveloppe zero-sum (style Necto/Nexto). En 1v1, selflessness n'a pas d'effet
    (pas de coéquipier) ; opp_coef rend le reward zero-sum vs l'adversaire :
    ce que TikshiV1 gagne, l'adversaire le perd.
    """
    return DistributeRewardsWrapper(
        reward_fn,
        selflessness=0.0,
        team_coef=1.0,
        opp_coef=opp_punish,
    )


def build_reward_fn(phase: int) -> RewardFunction:
    """Construit la fonction de récompense SOL-FIRST pour la phase donnée."""
    opp_punish = OPP_PUNISH_BY_PHASE[phase]

    # ── PHASE 1 — Bases (toucher la balle, ne pas oublier de sauter) ──────────
    # Pas de goal reward écrasant ; touch fort ; air léger anti-forget-jump.
    if phase == 1:
        return CombinedReward(
            (FaceBallReward(),                                       1.0),
            (SpeedTowardBallReward(),                                8.0),
            (VelocityBallToGoalReward(),                             6.0),
            (AdvancedTouchReward(touch_weight=1.0, accel_weight=1.5), 50.0),
            (SpeedflipKickoffReward(touch_bonus=1.0, speed_coef=0.5),  25.0),
            (SaveBoostReward(),                                      1.5),
            (BoostPickupReward(),                                    3.0),
            (InAirReward(),                                          0.15),
            (GoalDistancePotentialReward(gamma=1.0),                25.0),
            (GoalReward(),                                          50.0),
        )

    # ── PHASE 2 — Domination SOL SSL ──────────────────────────────────────────
    if phase == 2:
        zero_sum_rewards = _zero_sum(CombinedReward(
            (SaveReward(),                                          50.0),
            (DemoReward(attacker_reward=1.0, victim_punishment=1.0,
                        bump_acceleration_reward=0.3),              12.0),
            (GoalReward(),                                          60.0),
        ), opp_punish)
        individual_rewards = CombinedReward(
            (FaceBallReward(),                                       1.0),
            (SpeedTowardBallReward(),                                5.0),
            (VelocityBallToGoalReward(),                            10.0),
            (AdvancedTouchReward(touch_weight=0.5, accel_weight=2.0), 45.0),
            (DribbleReward(),                                       25.0),
            (FlickReward(),                                         30.0),
            (PerfectFlickReward(),                                  20.0),
            (ChallengeReward(),                                     15.0),
            (SpeedflipKickoffReward(touch_bonus=1.0, speed_coef=0.5,
                                    flip_bonus=0.4, supersonic_bonus=0.4),  25.0),
            (RecoveryReward(),                                       8.0),
            (WavedashReward(),                                      10.0),
            (DefendGoalReward(),                                     8.0),
            (BehindBallReward(),                                     2.0),
            (SaveBoostReward(),                                      2.0),
            (BoostPickupReward(),                                    4.0),
            (InAirReward(),                                          0.10),
            (GoalDistancePotentialReward(gamma=1.0),                30.0),
        )
        return CombinedReward((zero_sum_rewards, 1.0), (individual_rewards, 1.0))

    # ── PHASE 3 — Sol maîtrisé + INTRO aérienne (sol domine toujours) ─────────
    if phase == 3:
        zero_sum_rewards = _zero_sum(CombinedReward(
            (SaveReward(),                                          60.0),
            (DemoReward(attacker_reward=1.0, victim_punishment=1.0,
                        bump_acceleration_reward=0.3),              16.0),
            (GoalReward(),                                          80.0),
        ), opp_punish)
        individual_rewards = CombinedReward(
            # — SOL (poids forts) —
            (SpeedTowardBallReward(),                                4.0),
            (VelocityBallToGoalReward(),                            10.0),
            (AdvancedTouchReward(touch_weight=0.3, accel_weight=2.0), 45.0),
            (DribbleReward(),                                       30.0),
            (FlickReward(),                                         35.0),
            (PerfectFlickReward(),                                  30.0),
            (CatchReward(),                                          6.0),
            (ChallengeReward(),                                     15.0),
            (SpeedflipKickoffReward(touch_bonus=1.0, speed_coef=0.4,
                                    flip_bonus=0.4, supersonic_bonus=0.4),  20.0),
            (RecoveryReward(),                                       8.0),
            (WavedashReward(),                                      10.0),
            (DefendGoalReward(),                                    10.0),
            (BehindBallReward(),                                     2.0),
            (SaveBoostReward(),                                      2.5),
            (BoostPickupReward(),                                    5.0),
            (GoalDistancePotentialReward(gamma=1.0),                30.0),
            # — AÉRIEN (introduction, plafonné SOUS le sol) —
            (AerialTouchReward(min_height=400.0),                   40.0),
            (AerialNavigationReward(),                              10.0),
            (AerialDistanceReward(touch_height_weight=0.8,
                                  car_distance_weight=0.8,
                                  ball_distance_weight=0.8),        25.0),
            (AirDribbleReward(),                                     5.0),
            (InAirReward(),                                          0.12),
        )
        return CombinedReward((zero_sum_rewards, 1.0), (individual_rewards, 1.0))

    # ── PHASE 4 — Mécas SSL (wall play, ceiling, air dribble, flip reset) ─────
    # Priorité SOL maintenue : aérien fort mais TOUJOURS sous le sol.
    if phase == 4:
        zero_sum_rewards = _zero_sum(CombinedReward(
            (SaveReward(),                                          70.0),
            (DemoReward(attacker_reward=1.0, victim_punishment=1.0,
                        bump_acceleration_reward=0.4),              24.0),
            (FlipResetReward(obtain_flip_weight=15.0, hit_ball_weight=30.0), 12.0),
            (GoalReward(),                                          90.0),
        ), opp_punish)
        individual_rewards = CombinedReward(
            # — SOL (reste dominant) —
            (SpeedTowardBallReward(),                                3.0),
            (VelocityBallToGoalReward(),                            10.0),
            (AdvancedTouchReward(touch_weight=0.2, accel_weight=2.0), 45.0),
            (DribbleReward(),                                       35.0),
            (FlickReward(),                                         40.0),
            (PerfectFlickReward(),                                  45.0),
            (CatchReward(),                                          6.0),
            (ChallengeReward(),                                     15.0),
            (SpeedflipKickoffReward(touch_bonus=1.0, speed_coef=0.35,
                                    flip_bonus=0.4, supersonic_bonus=0.4),  15.0),
            (RecoveryReward(),                                       8.0),
            (WavedashReward(),                                      10.0),
            (DefendGoalReward(),                                    12.0),
            (BehindBallReward(),                                     2.0),
            (SaveBoostReward(),                                      3.0),
            (BoostPickupReward(),                                    5.0),
            (GoalDistancePotentialReward(gamma=1.0),                30.0),
            # — AÉRIEN avancé (fort mais plafonné sous le sol) —
            (AerialTouchReward(min_height=400.0),                   50.0),
            (AerialNavigationReward(),                              12.0),
            (AerialDistanceReward(touch_height_weight=1.0,
                                  car_distance_weight=1.0,
                                  ball_distance_weight=1.0),        40.0),
            (AirDribbleReward(),                                    20.0),
            (CeilingShotReward(),                                   30.0),
            (FlipResetReward(obtain_flip_weight=15.0,
                             hit_ball_weight=30.0),                 15.0),
            (InAirReward(),                                          0.10),
        )
        return CombinedReward((zero_sum_rewards, 1.0), (individual_rewards, 1.0))

    # ── PHASE 5 — SSL Grind équilibré ────────────────────────────────────────
    zero_sum_rewards = _zero_sum(CombinedReward(
        (SaveReward(),                                             80.0),
        (DemoReward(attacker_reward=1.0, victim_punishment=1.0,
                    bump_acceleration_reward=0.5),                 30.0),
        (FlipResetReward(obtain_flip_weight=20.0, hit_ball_weight=40.0), 20.0),
        (GoalReward(),                                            100.0),
    ), opp_punish)
    individual_rewards = CombinedReward(
        # — SOL (toujours au-dessus de l'aérien) —
        (VelocityBallToGoalReward(),                              12.0),
        (AdvancedTouchReward(touch_weight=0.0, accel_weight=2.5),  50.0),
        (DribbleReward(),                                         40.0),
        (FlickReward(),                                           45.0),
        (PerfectFlickReward(),                                    55.0),
        (CatchReward(),                                            8.0),
        (ChallengeReward(),                                       18.0),
        (SpeedflipKickoffReward(touch_bonus=1.0, speed_coef=0.3,
                                flip_bonus=0.4, supersonic_bonus=0.4),  12.0),
        (RecoveryReward(),                                         8.0),
        (WavedashReward(),                                        10.0),
        (DefendGoalReward(),                                      15.0),
        (BehindBallReward(),                                       2.0),
        (SaveBoostReward(),                                        3.5),
        (BoostPickupReward(),                                      6.0),
        (GoalDistancePotentialReward(gamma=1.0),                  30.0),
        # — AÉRIEN à son apex, mais plafonné SOUS le sol —
        (AerialTouchReward(min_height=400.0),                     60.0),
        (AerialNavigationReward(),                                15.0),
        (AerialDistanceReward(touch_height_weight=1.2,
                              car_distance_weight=1.0,
                              ball_distance_weight=1.0),           50.0),
        (AirDribbleReward(),                                      30.0),
        (CeilingShotReward(),                                     40.0),
        (FlipResetReward(obtain_flip_weight=20.0,
                         hit_ball_weight=40.0),                   25.0),
        (InAirReward(),                                            0.10),
    )
    return CombinedReward((zero_sum_rewards, 1.0), (individual_rewards, 1.0))

# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMÈTRES
# ─────────────────────────────────────────────────────────────────────────────

def get_hyperparams(phase: int) -> dict:
    """Hyperparams PPO par phase (GPU)."""
    base = {
        "n_proc":               N_PROC,
        "ppo_batch_size":       100_000,
        "ts_per_iteration":     100_000,
        "exp_buffer_size":      300_000,
        "ppo_minibatch_size":   50_000,
        "ppo_epochs":           3,
        # ppo_ent_coef est géré automatiquement par EntCoefAutoPilot ; cette
        # valeur sert uniquement à l'init avant que le pilote prenne la main.
        "ppo_ent_coef":          0.01,
        "policy_layer_sizes":   list(NET_SIZE),
        "critic_layer_sizes":   list(NET_SIZE),
        "save_every_ts":        5_000_000,
        "n_checkpoints_to_keep": 50,
        "render":               RENDER,
        "render_delay":         RENDER_REALTIME_DELAY if RENDER else 0.0,
        "standardize_returns":  True,
        "standardize_obs":      False,
        "add_unix_timestamp":   False,
        "log_to_wandb":         False,
    }
    lr_by_phase = {1: 2e-4, 2: 2e-4, 3: 1e-4, 4: 8e-5, 5: 5e-5}
    base["policy_lr"] = lr_by_phase[phase]
    base["critic_lr"] = lr_by_phase[phase]

    if phase >= 3:
        base["ppo_batch_size"]     = 150_000
        base["ts_per_iteration"]   = 150_000
        base["exp_buffer_size"]    = 450_000
        base["ppo_minibatch_size"] = 50_000

    if phase >= 5:
        base["ppo_batch_size"]     = 200_000
        base["ts_per_iteration"]   = 200_000
        base["exp_buffer_size"]    = 600_000
        base["ppo_minibatch_size"] = 50_000

    return base

# ─────────────────────────────────────────────────────────────────────────────
# ENT_COEF SCHEDULE — référence de position dans l'entraînement
# ─────────────────────────────────────────────────────────────────────────────

_ENT_COEF_SCHEDULE = [
    (0,                  0.01),
    (33_000_000,         0.005),
    (50_000_000,         0.003),
    (77_000_000,         0.002),
    (120_000_000,        0.0015),
    (180_000_000,        0.001),
    (280_000_000,        0.0008),
    (500_000_000,        0.0008),   # transition phase 3 — on garde
    (550_000_000,        0.0006),
    (800_000_000,        0.0005),
    (1_200_000_000,      0.0004),
    (1_500_000_000,      0.0004),   # transition phase 4 — on garde
    (1_600_000_000,      0.0003),
    (2_500_000_000,      0.00025),
    (4_000_000_000,      0.0002),
    (5_000_000_000,      0.0002),   # transition phase 5 — on garde
    (5_200_000_000,      0.00015),
    (7_000_000_000,      0.0001),
]

_ENT_COEF_LADDER = (
    0.01, 0.005, 0.003, 0.002, 0.0015, 0.0012,
    0.001, 0.0008, 0.0006, 0.0005, 0.0004, 0.0003,
    0.00025, 0.0002, 0.00015, 0.0001, 0.00005,
)


def get_schedule_ent_coef(total_steps: int) -> float:
    """ppo_ent_coef de référence pour le nombre de steps actuel (init autopilot)."""
    coef = _ENT_COEF_SCHEDULE[0][1]
    for threshold, value in _ENT_COEF_SCHEDULE:
        if total_steps >= threshold:
            coef = value
    return coef


# ─────────────────────────────────────────────────────────────────────────────
# ENT_COEF AUTOPILOT — auto-ajustement basé sur les métriques PPO
# ─────────────────────────────────────────────────────────────────────────────
#
# RÈGLES MAÎTRESSES :
#   1. reward_mean est l'indicateur principal. Baisse 20M+ → remonter ent_coef.
#   2. "entropy ↘ + reward ↘" = convergence prématurée → remonter IMMÉDIATEMENT.
#   3. Transition de phase = 10M de patience (lockout), fenêtre vidée.
#   4. Jamais > 1 palier à la fois. 5. Jamais 2 changements en < 20M.
# ─────────────────────────────────────────────────────────────────────────────

class EntCoefAutoPilot:
    """Auto-ajusteur de ppo_ent_coef intégré dans la boucle PPO."""

    _WINDOW_STEPS       = 20_000_000
    _MIN_GAP            = 20_000_000
    _PHASE_LOCKOUT      = 10_000_000
    _MIN_ENTRIES        = 8

    _ENT_DELTA_FLAT     = 0.02
    _RMAX_DELTA_RED     = 0.05
    _KL_RED             = 0.0045
    _CLIP_RED           = 0.04

    _RMEAN_DROP_URGENT  = -0.08
    _ENT_DROP_FAST      = -0.30
    _ENT_DROP_POSTCHG   = -0.20
    _POSTCHG_WINDOW     = 10_000_000
    _POSTCHG_LOOKBACK   = 5_000_000

    def __init__(self, initial_coef: float, initial_step: int = 0, verbose: bool = True) -> None:
        self._coef             = self._snap(initial_coef)
        self._history          = []
        self._window           = []
        self._phase            = self._phase_of(initial_step)
        self._phase_start_step = self._phase_start_of(initial_step)
        self._last_change_step = initial_step
        self._verbose          = verbose
        if verbose:
            idx = self._ladder_idx(self._coef)
            print(f"[AutoPilot] Démarré | ent_coef={self._coef} "
                  f"| palier {idx}/{len(_ENT_COEF_LADDER)-1} "
                  f"| init_step={initial_step:,} | phase={self._phase}")

    def update(self, step, reward_mean, reward_max, entropy, kl, clip_frac, vf_loss) -> None:
        """Appeler après chaque iteration PPO."""
        new_phase = self._phase_of(step)
        if new_phase != self._phase:
            self._phase           = new_phase
            self._phase_start_step = self._phase_start_of(step)
            self._window.clear()
            if self._verbose:
                print(f"\n[AutoPilot] ══ Phase {new_phase} détectée à "
                      f"{step:,} steps. Lockout 10M. Fenêtre vidée. ══\n")

        entry = dict(step=step, rm=float(reward_mean), rmax=float(reward_max),
                     ent=float(entropy), kl=float(kl),
                     clip=float(clip_frac), vfl=float(vf_loss))
        self._window.append(entry)
        cutoff = step - self._WINDOW_STEPS
        self._window = [e for e in self._window if e["step"] >= cutoff]

        if len(self._window) < self._MIN_ENTRIES:
            return
        if (step - self._phase_start_step) < self._PHASE_LOCKOUT:
            return

        d_ent   = self._delta_avg("ent")
        d_rmean = self._delta_rel_avg("rm")

        since_change = step - self._last_change_step
        if 0 < since_change < self._POSTCHG_WINDOW:
            d_ent_recent = self._delta_since(step - self._POSTCHG_LOOKBACK, "ent")
            if d_ent_recent is not None and d_ent_recent < self._ENT_DROP_POSTCHG:
                self._change("UP", step, n_paliers=1,
                             reason=(f"POST-CHANGE entropy chute trop vite "
                                     f"Δent_{self._POSTCHG_LOOKBACK // 1_000_000}M="
                                     f"{d_ent_recent:+.3f} → remontée"))
                return

        if d_ent < self._ENT_DROP_FAST and d_rmean < self._RMEAN_DROP_URGENT:
            self._change("UP", step, n_paliers=1,
                         reason=(f"URGENCE entropy↘+reward↘ Δent={d_ent:+.3f} "
                                 f"Δreward_mean={d_rmean:+.3f} → convergence prématurée"))
            return

        if (step - self._last_change_step) < self._MIN_GAP:
            return

        if d_rmean < self._RMEAN_DROP_URGENT:
            self._change("UP", step, n_paliers=1,
                         reason=f"CAS1 reward_mean chute sur 20M Δ={d_rmean:+.3f}")
            return

        d_rmax    = self._delta_rmax()
        mean_kl   = self._mean("kl")
        mean_clip = self._mean("clip")

        red_a = abs(d_ent) < self._ENT_DELTA_FLAT
        red_b = d_rmax < self._RMAX_DELTA_RED
        red_c = (mean_kl < self._KL_RED) or (mean_clip < self._CLIP_RED)
        n_red = int(red_a) + int(red_b) + int(red_c)

        if n_red >= 2 and d_rmean > 0:
            n_paliers = 2 if n_red >= 3 else 1
            self._change("DOWN", step, n_paliers=n_paliers,
                         reason=(f"DecisionTree {n_red}/3 rouges "
                                 f"[A(ent)={red_a} B(rmax)={red_b} C(KL/clip)={red_c}] "
                                 f"Δent={d_ent:+.3f} Δrmax={d_rmax:+.3f} "
                                 f"KL={mean_kl:.5f} clip={mean_clip:.4f}"))
            return

        if n_red == 0 and d_rmean < self._RMEAN_DROP_URGENT:
            self._change("UP", step, n_paliers=1,
                         reason=(f"Tout vert MAIS reward_mean baisse Δ={d_rmean:+.3f} "
                                 f"→ ent_coef trop bas, ré-exploration nécessaire"))
            return

        if self._verbose:
            print(f"[AutoPilot] HOLD | step={step:,} phase={self._phase} "
                  f"coef={self._coef} rouges={n_red}/3 | "
                  f"Δrm={d_rmean:+.3f} Δent={d_ent:+.3f} Δrmax={d_rmax:+.3f}")

    def get_coef(self) -> float:
        return self._coef

    def get_history(self):
        return list(self._history)

    def _change(self, direction: str, step: int, n_paliers: int, reason: str) -> None:
        old = self._coef
        idx = self._ladder_idx(old)
        if direction == "UP":
            nidx = max(0, idx - n_paliers)
        else:
            nidx = min(len(_ENT_COEF_LADDER) - 1, idx + n_paliers)
        new = _ENT_COEF_LADDER[nidx]
        if new == old:
            if self._verbose:
                bord = "MAX exploration" if direction == "UP" else "MIN exploration"
                print(f"[AutoPilot] {direction} demandé mais déjà au bord ({bord}).")
            return
        self._coef             = new
        self._last_change_step = step
        record = dict(step=step, phase=self._phase, direction=direction,
                      old=old, new=new, n_paliers=n_paliers, reason=reason)
        self._history.append(record)
        if self._verbose:
            arrow = "▲ UP  " if direction == "UP" else "▼ DOWN"
            label = f"{n_paliers} palier" + ("s" if n_paliers > 1 else "")
            print(f"\n[AutoPilot] {arrow} ({label}) | {old} → {new} "
                  f"| step={step:,} | phase={self._phase}\n            {reason}\n")

    def _delta_avg(self, key: str) -> float:
        if len(self._window) < 4:
            return 0.0
        k = max(2, len(self._window) // 5)
        head = sum(e[key] for e in self._window[:k]) / k
        tail = sum(e[key] for e in self._window[-k:]) / k
        return tail - head

    def _delta_rel_avg(self, key: str) -> float:
        if len(self._window) < 4:
            return 0.0
        k = max(2, len(self._window) // 5)
        head = sum(e[key] for e in self._window[:k]) / k
        tail = sum(e[key] for e in self._window[-k:]) / k
        denom = abs(head) if abs(head) > 1e-6 else 1.0
        return (tail - head) / denom

    def _delta_rmax(self) -> float:
        if len(self._window) < 4:
            return 0.0
        half = len(self._window) // 2
        old_max = max(e["rm"] for e in self._window[:half])
        new_max = max(e["rm"] for e in self._window[half:])
        denom = abs(old_max) if abs(old_max) > 1e-6 else 1.0
        return (new_max - old_max) / denom

    def _delta_since(self, since_step: int, key: str):
        slice_ = [e for e in self._window if e["step"] >= since_step]
        if len(slice_) < 2:
            return None
        return slice_[-1][key] - slice_[0][key]

    def _mean(self, key: str) -> float:
        if not self._window:
            return 0.0
        return sum(e[key] for e in self._window) / len(self._window)

    @staticmethod
    def _phase_of(step: int) -> int:
        phase = 1
        for p, start in sorted(_PHASE_STARTS.items()):
            if step >= start:
                phase = p
        return phase

    @staticmethod
    def _phase_start_of(step: int) -> int:
        start = 0
        for p, s in sorted(_PHASE_STARTS.items()):
            if step >= s:
                start = s
        return start

    @staticmethod
    def _snap(coef: float) -> float:
        return min(_ENT_COEF_LADDER, key=lambda x: abs(x - coef))

    @staticmethod
    def _ladder_idx(coef: float) -> int:
        return min(range(len(_ENT_COEF_LADDER)),
                   key=lambda i: abs(_ENT_COEF_LADDER[i] - coef))


# ─────────────────────────────────────────────────────────────────────────────
# BOUCLE AUTOPILOTÉE — remplace learner.learn()
# ─────────────────────────────────────────────────────────────────────────────

def run_with_autopilot(learner, pilot: EntCoefAutoPilot, timestep_limit: int,
                       log_every_n: int = 10) -> None:
    """
    Réplique la boucle interne de rlgym-ppo avec un hook autopilot après chaque
    itération PPO (pilot.update() + hot-swap de ent_coef). Comportement identique
    à learner.learn() : collecte, buffer, update, log, save, KBHit (p/c/q).
    """
    import time
    from rlgym_ppo.util import reporting

    try:
        from rlgym_ppo.util import KBHit
        kb = KBHit()
        kbhit_available = True
        print("Press (p) to pause (c) to checkpoint, (q) to checkpoint "
              "and quit (after next iteration)\n")
    except Exception:
        kb = None
        kbhit_available = False

    path = _find_ent_coef_path(learner)
    if path is not None:
        print(f"[AutoPilot] Chemin ent_coef détecté : learner.{path}")
    else:
        print("[AutoPilot] ATTENTION : aucun chemin ent_coef trouvé. "
              "L'autopilot décidera mais ne pourra PAS appliquer.")

    print(f"[AutoPilot] Boucle autopilotée | limite={timestep_limit:,} | "
          f"window={pilot._WINDOW_STEPS:,} | min_gap={pilot._MIN_GAP:,}")

    iteration = 0
    quit_requested = False

    while learner.agent.cumulative_timesteps < timestep_limit:
        epoch_start = time.perf_counter()
        report = {}

        (experience, collected_metrics, steps_collected,
         collection_time) = learner.agent.collect_timesteps(learner.ts_per_epoch)

        if learner.metrics_logger is not None:
            learner.metrics_logger.report_metrics(
                collected_metrics, learner.wandb_run,
                learner.agent.cumulative_timesteps,
            )

        learner.add_new_experience(experience)

        ppo_report = learner.ppo_learner.learn(learner.experience_buffer)
        epoch_stop = time.perf_counter()
        epoch_time = epoch_stop - epoch_start

        report.update(ppo_report)
        if learner.epoch < 1:
            report["Value Function Loss"] = float("nan")
        report["Cumulative Timesteps"]       = learner.agent.cumulative_timesteps
        report["Total Iteration Time"]       = epoch_time
        report["Timesteps Collected"]        = steps_collected
        report["Timestep Collection Time"]   = collection_time
        report["Timestep Consumption Time"]  = epoch_time - collection_time
        report["Collected Steps per Second"] = steps_collected / max(collection_time, 1e-9)
        report["Overall Steps per Second"]   = steps_collected / max(epoch_time, 1e-9)

        learner.ts_since_last_save += steps_collected
        if learner.agent.average_reward is not None:
            report["Policy Reward"] = learner.agent.average_reward
        else:
            report["Policy Reward"] = float("nan")

        step = int(report["Cumulative Timesteps"])
        rm   = report.get("Policy Reward")
        ent  = report.get("Policy Entropy")
        kl   = report.get("Mean KL Divergence")
        clip = report.get("SB3 Clip Fraction")
        vfl  = report.get("Value Function Loss")

        def _is_num(v):
            return isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))

        metrics_ok = all(_is_num(v) for v in (rm, ent, kl, clip, vfl))
        if metrics_ok:
            pilot.update(step=step, reward_mean=float(rm), reward_max=float(rm),
                         entropy=float(ent), kl=float(kl),
                         clip_frac=float(clip), vf_loss=float(vfl))
            _inject_ent_coef(learner, pilot.get_coef())
            if iteration > 0 and iteration % log_every_n == 0:
                _log_autopilot_status(step, pilot)
        else:
            if iteration < 5:
                missing = [name for name, v in (
                    ("Policy Reward", rm), ("Policy Entropy", ent),
                    ("Mean KL Divergence", kl), ("SB3 Clip Fraction", clip),
                    ("Value Function Loss", vfl),
                ) if not _is_num(v)]
                print(f"[AutoPilot] it={iteration} démarrage, "
                      f"métriques pas prêtes : {missing}")

        reporting.report_metrics(loggable_metrics=report, debug_metrics=None,
                                 wandb_run=learner.wandb_run)
        report.clear()
        ppo_report.clear()

        if "cuda" in str(learner.device):
            import torch
            torch.cuda.empty_cache()

        if kbhit_available and kb.kbhit():
            c = kb.getch()
            if c == 'p':
                print("Paused, press any key to resume")
                while True:
                    if kb.kbhit():
                        break
            if c in ('c', 'q'):
                learner.save(learner.agent.cumulative_timesteps)
                if c == 'q':
                    quit_requested = True
            if c in ('c', 'p'):
                print("Resuming...\n")

        if learner.ts_since_last_save >= learner.save_every_ts:
            learner.save(learner.agent.cumulative_timesteps)
            learner.ts_since_last_save = 0

        learner.epoch += 1
        iteration += 1

        if quit_requested:
            print("[AutoPilot] Quit clavier demandé. Sortie après save.")
            break

    # Force-save final pour franchir le seuil de phase (sinon relance dans la
    # même phase → boucle infinie). Idempotent.
    if learner.ts_since_last_save > 0:
        print(f"[AutoPilot] Force-save final "
              f"({learner.ts_since_last_save:,} steps depuis dernier save)...")
        try:
            learner.save(learner.agent.cumulative_timesteps)
            learner.ts_since_last_save = 0
        except Exception as exc:
            print(f"[AutoPilot] Force-save échoué : {exc}. "
                  "Le checkpoint précédent reste valide.")

    print(f"[AutoPilot] Boucle terminée à {learner.agent.cumulative_timesteps:,} steps "
          f"(limite={timestep_limit:,}).")


def _find_ent_coef_path(learner):
    """Renvoie le path (dotted) où ent_coef est accessible, ou None."""
    for path in ("ppo_learner.ent_coef", "agent.ent_coef", "ppo.ent_coef", "ent_coef"):
        parts = path.split(".")
        obj = learner
        ok = True
        for part in parts[:-1]:
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok and hasattr(obj, parts[-1]):
            return path
    return None


def _inject_ent_coef(learner, new_coef: float) -> None:
    """Injecte new_coef dans le PPO agent (path rlgym-ppo : ppo_learner.ent_coef)."""
    for path in ("ppo_learner.ent_coef", "agent.ent_coef", "ppo.ent_coef", "ent_coef"):
        parts = path.split(".")
        obj   = learner
        ok    = True
        for part in parts[:-1]:
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if not ok or not hasattr(obj, parts[-1]):
            continue
        cur = float(getattr(obj, parts[-1]))
        if abs(cur - new_coef) > 1e-10:
            setattr(obj, parts[-1], new_coef)
        return


def _log_autopilot_status(step: int, pilot: EntCoefAutoPilot) -> None:
    w = pilot._window
    if len(w) < 2:
        return
    rm_old  = w[0]["rm"];  rm_new  = w[-1]["rm"]
    ent_old = w[0]["ent"]; ent_new = w[-1]["ent"]
    span_M  = (w[-1]["step"] - w[0]["step"]) / 1_000_000
    print(f"[AutoPilot] step={step:,} | coef={pilot.get_coef()} | "
          f"window={len(w)} entrées ({span_M:.1f}M) | "
          f"reward_mean {rm_old:.0f}→{rm_new:.0f} | "
          f"entropy {ent_old:.3f}→{ent_new:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# EnvBuilder — classe picklable
# ─────────────────────────────────────────────────────────────────────────────


class EnvBuilder:
    """Callable picklable qui construit l'env 1v1 pour une phase donnée."""
    def __init__(self, phase: int):
        self.phase = phase

    def __call__(self):
        from rlgym.api import RLGym
        from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
        from rlgym.rocket_league.done_conditions import (
            GoalCondition, NoTouchTimeoutCondition, TimeoutCondition, AnyCondition
        )
        from rlgym.rocket_league.obs_builders import DefaultObs
        from rlgym.rocket_league.sim import RocketSimEngine
        from rlgym.rocket_league import common_values
        from rlgym_ppo.util import RLGymV2GymWrapper

        no_touch_to = {1: 30, 2: 20, 3: 15, 4: 12, 5: 10}[self.phase]

        action_parser    = RepeatAction(LookupTableAction(), repeats=8)
        termination_cond = GoalCondition()
        truncation_cond  = AnyCondition(
            NoTouchTimeoutCondition(timeout_seconds=no_touch_to),
            TimeoutCondition(timeout_seconds=300),
        )
        reward_fn     = build_reward_fn(self.phase)
        state_mutator = build_state_mutator(self.phase)

        # [1V1] zero_padding=1 : 1 self + 0 ally (padding) + 1 adversaire.
        # obs_dim = 52 + 20 * zero_padding * 2 = 92 dims (suffisant en 1v1).
        obs_builder = DefaultObs(
            zero_padding=1,
            pos_coef=np.asarray([
                1 / common_values.SIDE_WALL_X,
                1 / common_values.BACK_NET_Y,
                1 / common_values.CEILING_Z,
            ]),
            ang_coef     = 1 / np.pi,
            lin_vel_coef = 1 / common_values.CAR_MAX_SPEED,
            ang_vel_coef = 1 / common_values.CAR_MAX_ANG_VEL,
            boost_coef   = 1 / 100.0,
        )

        renderer = RocketSimVisRenderer() if RENDER else None

        env = RLGym(
            state_mutator    = state_mutator,
            obs_builder      = obs_builder,
            action_parser    = action_parser,
            reward_fn        = reward_fn,
            termination_cond = termination_cond,
            truncation_cond  = truncation_cond,
            transition_engine= RocketSimEngine(),
            renderer         = renderer,
        )
        return RLGymV2GymWrapper(env)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN — BOUCLE AUTOMATIQUE DE PHASES
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    from rlgym_ppo import Learner

    if DEVICE == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                print("╔══════════════════════════════════════════════════════════╗")
                print("║ ATTENTION : DEVICE='cuda' demandé mais CUDA INDISPONIBLE  ║")
                print("║ Fallback automatique sur CPU pour cette session.          ║")
                print("╚══════════════════════════════════════════════════════════╝")
                effective_device = "cpu"
            else:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"[CUDA] GPU détecté : {gpu_name} ({gpu_mem:.1f} GB VRAM)")
                effective_device = "cuda"
        except ImportError:
            print("[WARN] PyTorch non installé → fallback CPU.")
            effective_device = "cpu"
    else:
        effective_device = DEVICE

    print("=" * 70)
    print(f"  BOT 1V1 : {PROJECT_NAME}  (sol-first SSL — rewards Necto/Nexto/ZealanL)")
    print(f"  Device : {effective_device} | N_proc : {N_PROC} | Render : {RENDER}")
    print("=" * 70)

    while True:
        total_steps = get_total_steps_from_checkpoints(CHECKPOINT_FOLDER)
        phase       = get_phase(total_steps)
        hp          = get_hyperparams(phase)

        init_ent_coef = get_schedule_ent_coef(total_steps)
        hp["ppo_ent_coef"] = init_ent_coef

        timestep_limit = PHASE_THRESHOLDS[phase]

        chk_folder, chk_steps = find_latest_valid_checkpoint(CHECKPOINT_FOLDER)
        if chk_steps > total_steps:
            total_steps = chk_steps
        if chk_steps > 0 and chk_steps < total_steps:
            total_steps = chk_steps
            phase       = get_phase(total_steps)
            hp          = get_hyperparams(phase)
            init_ent_coef     = get_schedule_ent_coef(total_steps)
            hp["ppo_ent_coef"] = init_ent_coef
            timestep_limit    = PHASE_THRESHOLDS[phase]

        print(f"\n{'─'*70}")
        print(f"  {PHASE_NAMES[phase]}")
        print(f"  Steps : {total_steps:,} | Limite : {timestep_limit:,}")
        print(f"  LR : {hp['policy_lr']:.0e} | opp_punish : {OPP_PUNISH_BY_PHASE[phase]:.2f}")
        print(f"  [AUTOPILOT] ent_coef initial (schedule) : {init_ent_coef}")
        print(f"  Checkpoint : {chk_folder or 'Nouveau run'}")
        print(f"{'─'*70}\n")

        learner = Learner(
            EnvBuilder(phase),
            n_proc                  = hp["n_proc"],
            min_inference_size      = hp["n_proc"],
            ppo_batch_size          = hp["ppo_batch_size"],
            ts_per_iteration        = hp["ts_per_iteration"],
            exp_buffer_size         = hp["exp_buffer_size"],
            ppo_minibatch_size      = hp["ppo_minibatch_size"],
            ppo_epochs              = hp["ppo_epochs"],
            ppo_ent_coef            = hp["ppo_ent_coef"],
            policy_layer_sizes      = hp["policy_layer_sizes"],
            critic_layer_sizes      = hp["critic_layer_sizes"],
            policy_lr               = hp["policy_lr"],
            critic_lr               = hp["critic_lr"],
            save_every_ts           = hp["save_every_ts"],
            n_checkpoints_to_keep   = hp["n_checkpoints_to_keep"],
            checkpoint_load_folder  = chk_folder,
            checkpoints_save_folder = CHECKPOINT_FOLDER,
            add_unix_timestamp      = hp["add_unix_timestamp"],
            render                  = hp["render"],
            render_delay            = hp["render_delay"],
            metrics_logger          = None,
            standardize_returns     = hp["standardize_returns"],
            standardize_obs         = hp["standardize_obs"],
            device                  = effective_device,
            log_to_wandb            = hp["log_to_wandb"],
            timestep_limit          = timestep_limit,
        )

        pilot = EntCoefAutoPilot(
            initial_coef=init_ent_coef,
            initial_step=total_steps,
            verbose=True,
        )

        interrupted = False
        try:
            run_with_autopilot(learner, pilot, timestep_limit=timestep_limit)
        except KeyboardInterrupt:
            interrupted = True
            print("\n[CTRL+C] Arrêt demandé...")
        finally:
            try:
                learner.cleanup()
            except Exception:
                pass

        hist = pilot.get_history()
        if hist:
            print("\n[AUTOPILOT] Changements ent_coef cette session :")
            for rec in hist:
                arrow = "▲" if rec["direction"] == "UP" else "▼"
                print(f"  {arrow} step={rec['step']:>14,} | phase={rec['phase']} | "
                      f"{rec['old']} → {rec['new']} | {rec['reason']}")
        else:
            print("[AUTOPILOT] Aucun changement ent_coef cette session (HOLD).")

        if interrupted:
            print(f"\n[INFO] Arrêté à {total_steps:,} steps (phase {phase}). Au revoir.")
            break

        new_steps = get_total_steps_from_checkpoints(CHECKPOINT_FOLDER)
        new_phase = get_phase(new_steps)
        if new_phase != phase:
            print(f"\n{'='*70}")
            print(f"  >>> {PHASE_NAMES[new_phase]}")
            print(f"  Steps totaux : {new_steps:,}")
            print(f"{'='*70}\n")

        print("[INFO] Redémarrage automatique dans 3s...\n")
        time.sleep(3)
