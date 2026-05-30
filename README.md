# TikshiV1 — Bot Rocket League **1v1** (niveau SSL, sol-first)

TikshiV1 est un agent d'apprentissage par renforcement entraîné pour le **1v1**,
visant un niveau **SSL professionnel**. Contrairement à un bot spécialisé aérien,
TikshiV1 est **complet et équilibré** :

- **Sol ultra-dominant** : dribbles, flicks, speedflip kickoff, shadow defense,
  challenge/50, recovery, gestion du boost, shots puissants.
- **Aérien excellent** : aerials, wall play, ceiling shots, air dribbles, flip
  resets, double taps.
- **Règle d'or** : les mécaniques aériennes sont excellentes **mais ne sont
  jamais prioritaires** sur une solution plus efficace au sol. Concrètement, les
  rewards aériens sont *conditionnés* (balle haute / mur + boost) et *plafonnés*
  sous les rewards de sol, et il n'y a **aucune pénalité** pour rester au sol.

## Stack technique

- [RLGym 2.0](https://rlgym.org/) + [RocketSim](https://github.com/ZealanL/RocketSim) (simulation rapide)
- [rlgym-ppo](https://github.com/AechPro/rlgym-ppo) (PPO multi-process)
- [rlgym-tools](https://github.com/RLGym/rlgym-tools) (rewards avancés)
- PyTorch (GPU recommandé)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate            # Windows : .venv\Scripts\activate

pip install "rlgym[rl-rlviser]==2.0.1" rlgym-tools numpy
pip install git+https://github.com/AechPro/rlgym-ppo.git

# PyTorch GPU (adapte à ta version de CUDA, ex. cu121)
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Entraînement

```bash
python TikshiV1.py
```

Le script gère **tout automatiquement** :

1. Il lit les checkpoints dans `data/checkpoints/TikshiV1/` et **détecte la phase**.
2. Il entraîne jusqu'au seuil de la phase, puis **relance automatiquement** dans
   la phase suivante (aucune intervention manuelle).
3. L'**auto-pilot** (`EntCoefAutoPilot`) ajuste le coefficient d'entropie
   (`ppo_ent_coef`) en continu selon reward/entropy/KL/clip pour éviter la
   convergence prématurée.

Touches pendant l'entraînement : `p` = pause, `c` = checkpoint, `q` = checkpoint + quitter.

## Curriculum (5 phases, progression automatique)

| Phase | Steps | Focus |
|------:|-------|-------|
| 1 | 0 → 100M | Bases : toucher la balle, ne pas oublier de sauter, kickoff |
| 2 | 100M → 500M | **Domination sol SSL** : shots, dribbles, flicks, speedflip, saves, recovery, shadow defense, boost |
| 3 | 500M → 1.5B | Sol maîtrisé + **introduction aérienne** (sol majoritaire) |
| 4 | 1.5B → 5B | **Mécas SSL** : wall play, ceiling, air dribbles, flip resets, double taps (priorité sol maintenue) |
| 5 | 5B → ∞ | **SSL grind** : optimisation totale équilibrée, zero-sum maximal |

## Design des rewards (recherche)

- **RLGym-PPO-Guide (ZealanL)** : pas de goal reward écrasant ; touche pondérée
  par la force du contact ; air-touch = `min(air_time, height)` (vraies
  aériennes, pas de pop de mur plats) ; rewards **zero-sum** uniquement pour ce
  que l'adversaire doit empêcher (goals, demos, flip resets, powershots).
- **Lucy-SKG** (bat Necto/Nexto) : reward shaping utilitaire.
- **Necto / Nexto** : `opp_punish` (zero-sum) croissant + potential-based shaping
  (préserve la politique optimale).

## Visualiser le bot

Mettre `RENDER = True` **et** `N_PROC = 1` en haut de `TikshiV1.py`, puis lancer
[RocketSimVis](https://github.com/ZealanL/RocketSimVis) (écoute UDP `127.0.0.1:9273`).

## Configuration (en haut de `TikshiV1.py`)

| Constante | Rôle |
|-----------|------|
| `DEVICE` | `"cuda"` (GPU) ou `"cpu"` |
| `N_PROC` | nombre de processus RocketSim (CPU) |
| `NET_SIZE` | taille du réseau (def. `[512,512,512]` ; `[1024,1024,512]` = plafond + haut) |
| `RENDER` | `True` pour visualiser (avec `N_PROC=1`) |

> **Note** : atteindre le niveau SSL demande **des milliards de steps** et donc
> beaucoup de temps GPU. Le script est conçu pour tourner en continu et reprendre
> automatiquement après une interruption.
