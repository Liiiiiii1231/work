"""Stage 7：统一 Multi-start 多起点执行器。

本文件不修改任何单起点算法，只负责：
1. 对同一个方案使用多个随机 seed 重复运行；
2. 要求每个起点充分收敛；
3. 保存每个起点的 WSR、运行时间和迭代次数；
4. 按最终真实 WSR 选择最佳起点。

Stage 7 的 Multi-start 用于降低非凸算法对随机初始 ToMA 的敏感性。
它不是 Monte Carlo 统计实验；Monte Carlo 与参数扫描留到 Stage 8。
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Sequence

import numpy as np

from algorithm import run_joint_algorithm
from baselines import (
    run_fixed_toma_fp,
    run_fixed_toma_traditional,
    run_fpa_upa_fp,
    run_fpa_upa_traditional,
    run_rcg_toma_traditional,
)
from config import SystemConfig
from utils import check_resource_constraints


@dataclass
class StartRunResult:
    """保存一个随机起点的一次完整运行结果。"""

    seed: int
    result: Any
    final_wsr: float
    runtime_s: float
    iterations: int
    converged: bool


@dataclass
class MultiStartResult:
    """保存一个方案的全部 Multi-start 结果。"""

    name: str
    starts: list[StartRunResult]
    best_index: int

    @property
    def best_start(self) -> StartRunResult:
        return self.starts[self.best_index]

    @property
    def best_result(self) -> Any:
        return self.best_start.result

    @property
    def best_seed(self) -> int:
        return self.best_start.seed

    @property
    def best_wsr(self) -> float:
        return self.best_start.final_wsr

    @property
    def wsrs(self) -> np.ndarray:
        return np.asarray(
            [start.final_wsr for start in self.starts],
            dtype=float,
        )

    @property
    def total_runtime_s(self) -> float:
        return float(
            sum(start.runtime_s for start in self.starts)
        )


SingleStartRunner = Callable[[int], Any]
IterationGetter = Callable[[Any], int]


def run_multi_start(
    name: str,
    seeds: Sequence[int],
    run_single: SingleStartRunner,
    iteration_getter: IterationGetter,
    cfg: SystemConfig | None = None,
    require_convergence: bool = True,
    verbose: bool = False,
) -> MultiStartResult:
    """对一个单起点算法执行统一 Multi-start。"""

    normalized_seeds = tuple(int(seed) for seed in seeds)

    if len(normalized_seeds) == 0:
        raise ValueError("seeds must not be empty.")

    if len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("Multi-start seeds must be unique.")

    starts: list[StartRunResult] = []

    for seed in normalized_seeds:
        if verbose:
            print(f"  seed={seed} ...", end="", flush=True)

        start_time = perf_counter()
        result = run_single(seed)
        runtime_s = perf_counter() - start_time

        final_wsr = float(
            result.performance.weighted_sum_rate
        )

        if not np.isfinite(final_wsr):
            raise RuntimeError(
                f"{name}, seed={seed}: non-finite WSR."
            )

        converged = bool(result.converged)

        if require_convergence and not converged:
            raise RuntimeError(
                f"{name}, seed={seed}: did not converge."
            )

        if cfg is not None:
            check_resource_constraints(
                result.resources,
                cfg,
            )

        iterations = int(
            iteration_getter(result)
        )

        starts.append(
            StartRunResult(
                seed=seed,
                result=result,
                final_wsr=final_wsr,
                runtime_s=float(runtime_s),
                iterations=iterations,
                converged=converged,
            )
        )

        if verbose:
            print(
                f" WSR={final_wsr:.10f}, "
                f"iterations={iterations}, "
                f"time={runtime_s:.3f}s"
            )

    wsr_values = np.asarray(
        [start.final_wsr for start in starts],
        dtype=float,
    )

    best_index = int(
        np.argmax(wsr_values)
    )

    return MultiStartResult(
        name=name,
        starts=starts,
        best_index=best_index,
    )


def run_stage7_multistart(
    cfg: SystemConfig,
    seeds: Sequence[int],
    verbose: bool = False,
    require_convergence: bool = True,
) -> dict[str, MultiStartResult]:
    """对五个 baseline 和 Proposed 执行统一 Multi-start。

    B1、B2、B4、Proposed 使用相同 seed 重新创建独立 RNG，
    因而 ToMA 方案获得相同的随机初始化预算。

    B3、B5 使用固定 UPA。Stage 7 仍按全部 seed 重复运行，
    用于验证当前固定阵列方案是否对 seed 不敏感；Stage 8
    大规模 Monte Carlo 时可对已验证的固定方案只运行一次。
    """

    seed_tuple = tuple(int(seed) for seed in seeds)

    results: dict[str, MultiStartResult] = {}

    results["b1"] = run_multi_start(
        name="B1: RCG-ToMA + Traditional Resource Design",
        seeds=seed_tuple,
        run_single=lambda seed: run_rcg_toma_traditional(
            cfg,
            rng=np.random.default_rng(seed),
            verbose=False,
        ),
        iteration_getter=lambda result: result.iterations,
        cfg=cfg,
        require_convergence=require_convergence,
        verbose=verbose,
    )

    results["b2"] = run_multi_start(
        name="B2: Fixed-ToMA + FP Resource Optimization",
        seeds=seed_tuple,
        run_single=lambda seed: run_fixed_toma_fp(
            cfg,
            rng=np.random.default_rng(seed),
            verbose=False,
        ),
        iteration_getter=lambda result: result.iterations,
        cfg=cfg,
        require_convergence=require_convergence,
        verbose=verbose,
    )

    results["b3"] = run_multi_start(
        name="B3: FPA-UPA + FP Resource Optimization",
        seeds=seed_tuple,
        run_single=lambda seed: run_fpa_upa_fp(
            cfg,
            rng=np.random.default_rng(seed),
            verbose=False,
        ),
        iteration_getter=lambda result: result.iterations,
        cfg=cfg,
        require_convergence=require_convergence,
        verbose=verbose,
    )

    results["b4"] = run_multi_start(
        name="B4: Fixed-ToMA + Traditional Resource Design",
        seeds=seed_tuple,
        run_single=lambda seed: run_fixed_toma_traditional(
            cfg,
            rng=np.random.default_rng(seed),
        ),
        iteration_getter=lambda result: result.iterations,
        cfg=cfg,
        require_convergence=require_convergence,
        verbose=verbose,
    )

    results["b5"] = run_multi_start(
        name="B5: FPA-UPA + Traditional Resource Design",
        seeds=seed_tuple,
        run_single=lambda seed: run_fpa_upa_traditional(
            cfg,
        ),
        iteration_getter=lambda result: result.iterations,
        cfg=cfg,
        require_convergence=require_convergence,
        verbose=verbose,
    )

    results["proposed"] = run_multi_start(
        name="Proposed: RCG-ToMA + FP Resource Optimization",
        seeds=seed_tuple,
        run_single=lambda seed: run_joint_algorithm(
            cfg,
            rng=np.random.default_rng(seed),
            verbose=False,
        ),
        iteration_getter=lambda result: result.outer_iterations,
        cfg=cfg,
        require_convergence=require_convergence,
        verbose=verbose,
    )

    return results
