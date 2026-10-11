# ToMA-ISAC 清理前审查（尚未执行归档或删除）

日期：2026-10-10（Asia/Shanghai）

用户消息未包含逐项脚本/结果清理清单。以下为工程扫描得到的候选范围，等待用户补充清单后核对执行。

## experiments 中的 31 个候选脚本

- `experiments/diagnose_stage10_resource_switch.py`
- `experiments/plot_stage10b_final_boundary_adaptive.py`
- `experiments/plot_stage10b_final_boundary_v4.py`
- `experiments/plot_stage10b_final_boundary_v5.py`
- `experiments/plot_stage10_anchor_extension.py`
- `experiments/plot_stage10_bidirectional.py`
- `experiments/plot_stage10_final_weights.py`
- `experiments/plot_stage10_formal.py`
- `experiments/plot_stage10_formal_20.py`
- `experiments/plot_stage10_pool_refinement.py`
- `experiments/plot_stage10_probe_proposed.py`
- `experiments/plot_stage10_selective_warmstart.py`
- `experiments/plot_stage10_targeted20.py`
- `experiments/plot_stage10_tolerance_probe.py`
- `experiments/plot_stage10_transition_probe.py`
- `experiments/plot_stage10_weighted_coarse.py`
- `experiments/run_stage10_anchor_extension.py`
- `experiments/run_stage10_b6_second_pass.py`
- `experiments/run_stage10_bidirectional.py`
- `experiments/run_stage10_final_weights.py`
- `experiments/run_stage10_formal.py`
- `experiments/run_stage10_formal_20.py`
- `experiments/run_stage10_pool_refinement.py`
- `experiments/run_stage10_probe_proposed.py`
- `experiments/run_stage10_selective_warmstart.py`
- `experiments/run_stage10_targeted20.py`
- `experiments/run_stage10_tolerance_probe.py`
- `experiments/run_stage10_transition_probe.py`
- `experiments/run_stage10_weighted_stabilized.py`
- `experiments/stage10_20_support.py`
- `experiments/validate_stage10_midpoints.py`

## 历史结果目录候选（待核对用户清单）

- `results/stage10/b6_second_pass_from_0p5/`
- `results/stage10/cold20_targeted_warm_dl114/`
- `results/stage10/cold_dense_tol1e4/`
- `results/stage10/figures/`
- `results/stage10/formal_14point_multiround/`
- `results/stage10/formal_20point_dl114_multiround/`
- `results/stage10/local_convergence_diagnosis/`
- `results/stage10/midpoint_feasibility_cold20_dl114/`
- `results/stage10/pool_wide14_tol1e4/`
- `results/stage10/selective_warmstart_tol1e4/`
- `results/stage10/tolerance_comparison_cold/`
- `results/stage10/transition_050_060_tol1e4/`

Stage 10 根目录 stage10_tradeoff.csv、stage10_tradeoff.npz、stage10_failures.csv 及工程根目录两个诊断脚本尚未纳入删除范围。

## 历史测试候选

- `tests/test_pool_refinement.py`：动态加载 run_stage10_pool_refinement.py
- `tests/test_stage10_bidirectional.py`：导入 run_stage10_bidirectional.py
- `tests/test_stage10_cold_dense.py`：断言旧 27 点网格、cold_dense 目录和旧 DL 噪声
- `tests/test_stage10_tradeoff.py`：断言旧 27 点网格，与现有 14 点入口不符
- `tests/test_rcs114_unified.py`：断言旧 DL=-70 dBm 噪声和包含零权重点

保留 tests/test_stage10a_warmstart.py：它直接回归验证核心算法仍提供的热启动接口，不依赖历史诊断脚本；不因名称包含 warmstart 就移除。

## 明确保留

- `experiments/run_stage10_cold20.py`
- `experiments/plot_stage10_cold20.py`
- `experiments/run_stage10_tradeoff.py`
- `experiments/plot_stage10_tradeoff.py`
- 全部 Stage 5～8 程序及完整 results/stage8/
- results/stage10/cold_wide20_dl114_tol1e4/
- results/stage10/cold_wide14_tol1e4/
- run_all_figures_continue.py（维持 14 点入口）
- 所有核心算法文件、所有 YAML 原样保留

## 已完成的清理前检查

- 扫描工程 Python 源码中的候选脚本模块名和文件名引用。正式程序未发现候选引用；仅上述两个历史测试引用候选脚本。
- Python 语法：87 个文件通过（内存 compile，不生成字节码）。
- 关键模块导入：12 个通过。
- 总入口 check_environment()：通过，14 点依赖与正式配置可用。
- 原始 20 点 CSV：4 个方案各 20 个唯一、完整权重点，共 80 行，通过脚本收敛与有限值校验。
- 基础测试：test_geometry.py、test_channels.py、test_metrics.py、test_smoke.py，共 5 passed。未执行会调用优化器的测试。
- plot_stage10_cold20.py --output-dir tmp/cleanup_preflight_20261010/figures：成功生成 PNG/PDF/SVG。
- 未重跑数值优化，未覆盖正式结果；本次新增仅 tmp/cleanup_preflight_20261010/ 中的审查日志、报告和验证图。
- 使用 D:/miniconda/envs/toma_isac/python.exe。默认 python 缺少 numpy，未安装或修改环境。

## 待执行

核对用户逐项清单 → 输出最终拟清理范围 → 复制至工程外独立归档目录 → 逐文件 SHA-256 核验 → 删除工作目录对应原文件 → 清理后语法/导入/非优化测试/绘图验证。
