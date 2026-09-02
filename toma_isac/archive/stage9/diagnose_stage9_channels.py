"""Stage 9B：检查 ToMA 几何与纯 LoS 信道实现是否与模型一致。"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from channels import build_channels
from config import SystemConfig, load_config
from geometry import build_geometry
from initialization import generate_feasible_endpoints
from outer_solver import sphere_retraction

CONFIG_PATH = ROOT / "configs" / "stage9_outer_debug.yaml"
RESULT_PATH = (
    ROOT
    / "results"
    / "stage9"
    / "channel_sensitivity_scene_8001_seed_2029.csv"
)

SCENE_SEED = 8001
START_SEED = 2029
POSITION_JITTER = np.array([1.5, 1.5, 0.25], dtype=float)
PERTURBATIONS = (1.0e-6, 1.0e-5, 1.0e-4)


def _make_position_scene_cfg(cfg: SystemConfig) -> SystemConfig:
    """只复现 scene 位置随机化，不引入任何额外 fading。"""

    rng = np.random.default_rng(SCENE_SEED)

    def perturb(points: np.ndarray) -> np.ndarray:
        delta = rng.uniform(
            low=-POSITION_JITTER,
            high=POSITION_JITTER,
            size=points.shape,
        )
        return np.asarray(points + delta, dtype=float)

    return replace(
        cfg,
        q_dl=perturb(cfg.q_dl),
        q_ul=perturb(cfg.q_ul),
        q_s=perturb(cfg.q_s),
    )


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(expected)), 1.0e-30)
    return float(np.linalg.norm(actual - expected) / denominator)


def _relative_change(new: np.ndarray, old: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(old)), 1.0e-30)
    return float(np.linalg.norm(new - old) / denominator)


def _require_close(
    name: str,
    actual: np.ndarray,
    expected: np.ndarray,
    rtol: float = 1.0e-10,
    atol: float = 1.0e-12,
) -> None:
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        error = _relative_error(actual, expected)
        raise AssertionError(f"{name} mismatch: relative error={error:.3e}")


def _unit_tangent(point: np.ndarray, cfg: SystemConfig) -> np.ndarray:
    """构造一个确定性的单位切向方向。"""

    basis = np.array([1.0, 0.0, 0.0])
    point_unit = point / np.linalg.norm(point)

    if abs(float(np.dot(point_unit, basis))) > 0.9:
        basis = np.array([0.0, 1.0, 0.0])

    tangent = basis - point_unit * np.dot(point_unit, basis)
    tangent_norm = np.linalg.norm(tangent)

    if tangent_norm <= 1.0e-14:
        raise RuntimeError("Failed to construct tangent direction.")

    return tangent / tangent_norm


def _manual_channel_checks(cfg: SystemConfig, geometry) -> None:
    """逐项用公式重建 h/f/g/H_SI^0/H_RSI。"""

    channels = build_channels(geometry, cfg)

    fractions = (
        np.arange(1, cfg.n_cable + 1, dtype=float)
        / cfg.n_cable
    )
    all_positions = geometry.endpoints[:, None, :] * fractions[None, :, None]

    expected_tx = all_positions[: cfg.n_tx_uav].reshape(cfg.n_tx, 3)
    expected_rx = all_positions[cfg.n_tx_uav :].reshape(cfg.n_rx, 3)

    _require_close("tx_positions", geometry.tx_positions, expected_tx)
    _require_close("rx_positions", geometry.rx_positions, expected_rx)

    for k in range(cfg.k_dl):
        distance = np.linalg.norm(
            geometry.tx_positions - cfg.q_dl[k][None, :],
            axis=1,
        )
        expected = cfg.alpha_dl[k] * np.exp(1j * cfg.k0 * distance)
        _require_close(f"h[{k}]", channels.h[k], expected)

    for j in range(cfg.j_ul):
        distance = np.linalg.norm(
            geometry.rx_positions - cfg.q_ul[j][None, :],
            axis=1,
        )
        expected = cfg.alpha_ul[j] * np.exp(1j * cfg.k0 * distance)
        _require_close(f"f[{j}]", channels.f[j], expected)

    for ell in range(cfg.l_target):
        d_tx = np.linalg.norm(
            geometry.tx_positions - cfg.q_s[ell][None, :],
            axis=1,
        )
        d_rx = np.linalg.norm(
            geometry.rx_positions - cfg.q_s[ell][None, :],
            axis=1,
        )

        a_t = np.exp(1j * cfg.k0 * d_tx)
        a_r = np.exp(1j * cfg.k0 * d_rx)
        alpha_s = cfg.xi_s[ell] * np.sqrt(
            cfg.beta_t_s[ell] * cfg.beta_r_s[ell]
        )
        expected = alpha_s * np.outer(a_r.conj(), a_t.conj())
        _require_close(f"g[{ell}]", channels.g[ell], expected)

    difference = (
        geometry.rx_positions[:, None, :]
        - geometry.tx_positions[None, :, :]
    )
    distance = np.linalg.norm(difference, axis=2)

    expected_si0 = (
        cfg.wavelength
        / (4.0 * np.pi * distance)
        * np.exp(-1j * cfg.k0 * distance)
    )
    expected_rsi = np.sqrt(cfg.rho_si) * expected_si0

    _require_close("h_si0", channels.h_si0, expected_si0)
    _require_close("h_rsi", channels.h_rsi, expected_rsi)

    # 纯 LoS 当前模型：公共幅度增益相同，逐阵元差别只来自球面波相位。
    for k in range(cfg.k_dl):
        expected_mag = np.full(cfg.n_tx, abs(cfg.alpha_dl[k]))
        _require_close(f"|h[{k}]|", np.abs(channels.h[k]), expected_mag)

    for j in range(cfg.j_ul):
        expected_mag = np.full(cfg.n_rx, abs(cfg.alpha_ul[j]))
        _require_close(f"|f[{j}]|", np.abs(channels.f[j]), expected_mag)

    ratio = channels.h_rsi / channels.h_si0
    expected_ratio = np.full_like(ratio, np.sqrt(cfg.rho_si))
    _require_close("h_rsi / h_si0", ratio, expected_ratio)

    print("[PASS] Formula reconstruction")
    print(f"  rho_si = {cfg.rho_si:.1e}")
    print(f"  sqrt(rho_si) = {np.sqrt(cfg.rho_si):.3e}")
    print(f"  min direct-SI distance = {np.min(distance):.6e} m")
    print(f"  max direct-SI distance = {np.max(distance):.6e} m")


def _dependency_and_sensitivity_checks(cfg: SystemConfig, geometry) -> None:
    """检查 Tx/Rx 端点移动时哪些信道应变化，并输出灵敏度。"""

    base = build_channels(geometry, cfg)
    records: list[dict[str, float | int | str]] = []

    cases = (
        (0, "Tx"),
        (cfg.n_tx_uav, "Rx"),
    )

    for endpoint_index, group in cases:
        point = geometry.endpoints[endpoint_index]
        direction = _unit_tangent(point, cfg)

        for epsilon in PERTURBATIONS:
            candidate = sphere_retraction(
                point,
                direction,
                epsilon,
                cfg,
            )

            endpoints = geometry.endpoints.copy()
            endpoints[endpoint_index] = candidate

            moved_geometry = build_geometry(
                endpoints,
                cfg,
                check_constraints=False,
            )
            moved = build_channels(moved_geometry, cfg)

            dh = _relative_change(moved.h, base.h)
            df = _relative_change(moved.f, base.f)
            dg = _relative_change(moved.g, base.g)
            dsi0 = _relative_change(moved.h_si0, base.h_si0)
            drsi = _relative_change(moved.h_rsi, base.h_rsi)

            records.append(
                {
                    "endpoint_index": endpoint_index,
                    "group": group,
                    "epsilon_m": epsilon,
                    "rel_change_h": dh,
                    "rel_change_f": df,
                    "rel_change_g": dg,
                    "rel_change_h_si0": dsi0,
                    "rel_change_h_rsi": drsi,
                }
            )

            if group == "Tx":
                if df > 1.0e-14:
                    raise AssertionError(
                        f"Tx move unexpectedly changed UL channel f: {df:.3e}"
                    )
                if min(dh, dg, dsi0, drsi) <= 1.0e-12:
                    raise AssertionError(
                        "Tx move did not change all expected dependent channels."
                    )
            else:
                if dh > 1.0e-14:
                    raise AssertionError(
                        f"Rx move unexpectedly changed DL channel h: {dh:.3e}"
                    )
                if min(df, dg, dsi0, drsi) <= 1.0e-12:
                    raise AssertionError(
                        "Rx move did not change all expected dependent channels."
                    )

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print("[PASS] Geometry-to-channel dependency")
    print(f"[PASS] Sensitivity CSV -> {RESULT_PATH}")

    for record in records:
        print(
            "  "
            f"{record['group']} eps={record['epsilon_m']:.0e}: "
            f"dh={record['rel_change_h']:.3e}, "
            f"df={record['rel_change_f']:.3e}, "
            f"dg={record['rel_change_g']:.3e}, "
            f"dSI={record['rel_change_h_si0']:.3e}"
        )


def main() -> None:
    cfg = load_config(CONFIG_PATH)

    if not np.isclose(cfg.rho_si, 1.0e-10, rtol=0.0, atol=0.0):
        raise AssertionError("Stage 9 debug config must fix rho_si=1e-10.")

    if not np.isclose(cfg.finite_diff_delta, 1.0e-6):
        raise AssertionError("finite_diff_delta must remain 1e-6.")

    if cfg.plateau_window != 0 or cfg.tol_window != 0.0:
        raise AssertionError("Window plateau stop must be disabled in Stage 9.")

    scene_cfg = _make_position_scene_cfg(cfg)
    rng = np.random.default_rng(START_SEED)
    endpoints = generate_feasible_endpoints(scene_cfg, rng)
    geometry = build_geometry(endpoints, scene_cfg)

    print("Stage 9B channel model diagnostic")
    print(f"  scene seed = {SCENE_SEED}")
    print(f"  start seed = {START_SEED}")

    _manual_channel_checks(scene_cfg, geometry)
    _dependency_and_sensitivity_checks(scene_cfg, geometry)

    print("\nAll Stage 9B channel checks passed.")


if __name__ == "__main__":
    main()
