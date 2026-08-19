"""检查 rho_si -> H_RSI -> UL / sensing SINR 的数量级。"""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines import run_fixed_toma_fp
from channels import build_channels
from config import load_config
from metrics import compute_performance
from stage8 import (
    load_stage8_experiment_config,
    make_scene_config,
)


CONFIG_PATH = (
    ROOT
    / "configs"
    / "stage8_monte_carlo.yaml"
)

RHO_VALUES = (
    1.0e-10,
    1.0e-9,
    1.0e-8,
    1.0e-7,
    1.0e-6,
    1.0e-5,
    1.0e-4,
    1.0e-3,
)


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    experiment_cfg = load_stage8_experiment_config(
        CONFIG_PATH
    )

    # 固定一个 Monte Carlo 场景。
    scene_cfg = make_scene_config(
        cfg,
        scene_seed=8001,
        position_jitter=experiment_cfg.position_jitter,
    )

    # 先在 nominal rho_si 下得到一组固定 geometry/resources。
    reference_cfg = replace(
        scene_cfg,
        rho_si=1.0e-8,
    )

    result = run_fixed_toma_fp(
        reference_cfg,
        rng=np.random.default_rng(2026),
        verbose=False,
    )

    geometry = result.geometry
    resources = result.resources

    print("=" * 100)
    print("rho_si sensitivity check")
    print("=" * 100)

    print(
        f"Reference WSR = "
        f"{result.performance.weighted_sum_rate:.6f}"
    )

    print(
        f"||H_SI0||_F = "
        f"{np.linalg.norm(result.channels.h_si0):.6e}"
    )

    print()

    for rho_si in RHO_VALUES:
        test_cfg = replace(
            scene_cfg,
            rho_si=rho_si,
        )

        # geometry/resources 固定，只重新生成 H_RSI。
        channels = build_channels(
            geometry,
            test_cfg,
        )

        performance = compute_performance(
            channels,
            resources,
            test_cfg,
        )

        g_sum = np.sum(
            channels.g,
            axis=0,
        )

        ul_rsi_terms = []
        ul_echo_terms = []
        ul_combined_terms = []

        for j in range(test_cfg.j_ul):
            b_j = resources.b_ul[j]

            rsi_only = (
                np.linalg.norm(
                    b_j.conj()
                    @ channels.h_rsi
                    @ resources.q_matrix
                )
                ** 2
            )

            echo_only = (
                np.linalg.norm(
                    b_j.conj()
                    @ g_sum
                    @ resources.q_matrix
                )
                ** 2
            )

            combined = (
                np.linalg.norm(
                    b_j.conj()
                    @ (
                        g_sum
                        + channels.h_rsi
                    )
                    @ resources.q_matrix
                )
                ** 2
            )

            ul_rsi_terms.append(rsi_only)
            ul_echo_terms.append(echo_only)
            ul_combined_terms.append(combined)

        sensing_rsi_terms = []
        sensing_echo_terms = []
        sensing_combined_terms = []

        for ell in range(test_cfg.l_target):
            u_l = resources.u_s[ell]

            other_echo = (
                g_sum
                - channels.g[ell]
            )

            rsi_only = (
                np.linalg.norm(
                    u_l.conj()
                    @ channels.h_rsi
                    @ resources.q_matrix
                )
                ** 2
            )

            echo_only = (
                np.linalg.norm(
                    u_l.conj()
                    @ other_echo
                    @ resources.q_matrix
                )
                ** 2
            )

            combined = (
                np.linalg.norm(
                    u_l.conj()
                    @ (
                        other_echo
                        + channels.h_rsi
                    )
                    @ resources.q_matrix
                )
                ** 2
            )

            sensing_rsi_terms.append(rsi_only)
            sensing_echo_terms.append(echo_only)
            sensing_combined_terms.append(combined)

        print("-" * 100)

        print(
            f"rho_si = {rho_si:.1e}"
        )

        print(
            f"sqrt(rho_si) = "
            f"{np.sqrt(rho_si):.3e}"
        )

        print(
            f"||H_RSI||_F = "
            f"{np.linalg.norm(channels.h_rsi):.6e}"
        )

        print(
            "UL: "
            f"RSI-only={np.mean(ul_rsi_terms):.6e}, "
            f"echo-only={np.mean(ul_echo_terms):.6e}, "
            f"combined={np.mean(ul_combined_terms):.6e}"
        )

        print(
            "Sensing: "
            f"RSI-only={np.mean(sensing_rsi_terms):.6e}, "
            f"echo-only={np.mean(sensing_echo_terms):.6e}, "
            f"combined={np.mean(sensing_combined_terms):.6e}"
        )

        print(
            f"mean gamma_UL = "
            f"{np.mean(performance.gamma_ul):.6e}"
        )

        print(
            f"mean gamma_S = "
            f"{np.mean(performance.gamma_s):.6e}"
        )

        print(
            f"fixed-resource WSR = "
            f"{performance.weighted_sum_rate:.6f}"
        )


if __name__ == "__main__":
    main()