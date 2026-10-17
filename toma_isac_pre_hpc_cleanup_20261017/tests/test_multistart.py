"""测试 Stage 7 Multi-start 选择逻辑和六方案接口。"""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from config import load_config
from multistart import (
    run_multi_start,
    run_stage7_multistart,
)
from state import PerformanceResult, ResourceState


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "stage7_multistart.yaml"
)


@dataclass
class _DummyResult:
    """仅用于测试 Multi-start 选择逻辑。"""

    performance: PerformanceResult
    resources: ResourceState
    converged: bool
    iterations: int


def _dummy_result(
    wsr: float,
    converged: bool = True,
) -> _DummyResult:
    performance = PerformanceResult(
        gamma_dl=np.zeros(1),
        gamma_ul=np.zeros(1),
        gamma_s=np.zeros(1),
        rate_dl=np.zeros(1),
        rate_ul=np.zeros(1),
        rate_s=np.zeros(1),
        weighted_sum_rate=float(wsr),
    )

    resources = ResourceState(
        q_matrix=np.zeros((1, 1), dtype=np.complex128),
        q_ul=np.zeros(1),
        b_ul=np.ones((1, 1), dtype=np.complex128),
        u_s=np.ones((1, 1), dtype=np.complex128),
    )

    return _DummyResult(
        performance=performance,
        resources=resources,
        converged=converged,
        iterations=1,
    )


def test_multi_start_selects_largest_final_wsr() -> None:
    """验证最佳起点按最终真实 WSR 选择。"""

    values = {
        11: 3.0,
        12: 5.0,
        13: 4.0,
    }

    result = run_multi_start(
        name="dummy",
        seeds=(11, 12, 13),
        run_single=lambda seed: _dummy_result(values[seed]),
        iteration_getter=lambda item: item.iterations,
        cfg=None,
    )

    assert result.best_seed == 12
    assert np.isclose(result.best_wsr, 5.0)
    assert np.allclose(result.wsrs, [3.0, 5.0, 4.0])


def test_multi_start_rejects_duplicate_seed_and_nonconvergence() -> None:
    """验证 seed 唯一性和正式 Multi-start 的收敛要求。"""

    with pytest.raises(ValueError, match="unique"):
        run_multi_start(
            name="dummy",
            seeds=(1, 1),
            run_single=lambda seed: _dummy_result(1.0),
            iteration_getter=lambda item: item.iterations,
        )

    with pytest.raises(RuntimeError, match="did not converge"):
        run_multi_start(
            name="dummy",
            seeds=(1,),
            run_single=lambda seed: _dummy_result(
                1.0,
                converged=False,
            ),
            iteration_getter=lambda item: item.iterations,
        )


def test_stage7_all_schemes_use_same_seed_order() -> None:
    """快速验证六方案使用同一组 Stage 7 seed。"""

    cfg = load_config(CONFIG_PATH)

    # 只做接口 smoke test，不要求短迭代配置达到正式收敛。
    cfg = replace(
        cfg,
        max_inner_iter=2,
        max_outer_iter=1,
    )

    seeds = (31, 32)

    results = run_stage7_multistart(
        cfg,
        seeds=seeds,
        verbose=False,
        require_convergence=False,
    )

    assert set(results) == {
        "b1",
        "b2",
        "b3",
        "b4",
        "b5",
        "proposed",
    }

    for scheme_result in results.values():
        assert tuple(
            start.seed
            for start in scheme_result.starts
        ) == seeds

        assert np.all(
            np.isfinite(scheme_result.wsrs)
        )

    # B4 与其它随机 ToMA baseline 使用相同 seed，
    # 因而初始 ToMA 必须一致。
    assert np.allclose(
        results["b1"].starts[0].result.initial_endpoints,
        results["b2"].starts[0].result.initial_endpoints,
    )
    assert np.allclose(
        results["b1"].starts[0].result.initial_endpoints,
        results["b4"].starts[0].result.initial_endpoints,
    )

    # B5 完全固定，因此不同 seed 下应严格得到相同 WSR。
    assert np.allclose(
        results["b5"].wsrs,
        results["b5"].wsrs[0],
        rtol=0.0,
        atol=0.0,
    )
