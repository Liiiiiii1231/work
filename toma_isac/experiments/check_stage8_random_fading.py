"""Stage 8B-1：检查 scene-level 随机衰落的可复现性与固定性。"""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from stage8 import (
    load_stage8_experiment_config,
    make_scene_config,
)


CONFIG_PATH = ROOT / "configs" / "stage8_monte_carlo.yaml"


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    exp = load_stage8_experiment_config(CONFIG_PATH)

    seed = exp.scene_seed_start

    old_scene = make_scene_config(
        cfg,
        scene_seed=seed,
        position_jitter=exp.position_jitter,
        randomize_fading=False,
    )

    scene_a = make_scene_config(
        cfg,
        scene_seed=seed,
        position_jitter=exp.position_jitter,
        randomize_fading=exp.fading_enabled,
        fading_model=exp.fading_model,
    )
    scene_b = make_scene_config(
        cfg,
        scene_seed=seed,
        position_jitter=exp.position_jitter,
        randomize_fading=exp.fading_enabled,
        fading_model=exp.fading_model,
    )
    scene_c = make_scene_config(
        cfg,
        scene_seed=seed + 1,
        position_jitter=exp.position_jitter,
        randomize_fading=exp.fading_enabled,
        fading_model=exp.fading_model,
    )

    same_positions = (
        np.allclose(old_scene.q_dl, scene_a.q_dl)
        and np.allclose(old_scene.q_ul, scene_a.q_ul)
        and np.allclose(old_scene.q_s, scene_a.q_s)
    )

    same_fading = (
        np.allclose(scene_a.alpha_dl, scene_b.alpha_dl)
        and np.allclose(scene_a.alpha_ul, scene_b.alpha_ul)
        and np.allclose(scene_a.xi_s, scene_b.xi_s)
    )

    different_fading = (
        not np.allclose(scene_a.alpha_dl, scene_c.alpha_dl)
        and not np.allclose(scene_a.alpha_ul, scene_c.alpha_ul)
        and not np.allclose(scene_a.xi_s, scene_c.xi_s)
    )

    print("=" * 72)
    print("Stage 8B-1 random fading check")
    print("=" * 72)
    print(f"scene seed = {seed}")
    print(f"fading enabled = {exp.fading_enabled}")
    print(f"fading model = {exp.fading_model}")
    print(f"old/new scene positions unchanged = {same_positions}")
    print(f"same scene fading reproducible = {same_fading}")
    print(f"different scene fading changes = {different_fading}")
    print(f"alpha_dl = {scene_a.alpha_dl}")
    print(f"alpha_ul = {scene_a.alpha_ul}")
    print(f"xi_s = {scene_a.xi_s}")

    if not (same_positions and same_fading and different_fading):
        raise RuntimeError("Stage 8B-1 random fading check failed.")

    print("Stage 8B-1 random fading check passed.")


if __name__ == "__main__":
    main()
