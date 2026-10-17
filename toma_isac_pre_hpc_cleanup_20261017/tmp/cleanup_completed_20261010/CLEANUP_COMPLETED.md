# ToMA-ISAC Stage 10 清理完成报告

日期：2026-10-10（Asia/Shanghai）

## 结果

已按用户确认的清理前报告完成全部 48 项：31 个历史实验脚本、12 个历史结果目录、5 个旧测试。共 2,881 个文件（9,244,899 字节）已完整复制到工程外，保留原始目录结构；复制后和移除前逐文件核验 SHA-256，最终再次核验归档。

归档完整路径：`D:/Vs codeProjects/research work1/toma_isac_archive_20261010`

## 实际归档并移除的历史脚本

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

## 实际归档并移除的历史结果目录

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

## 实际归档并移除的旧测试

- `tests/test_pool_refinement.py`
- `tests/test_stage10_bidirectional.py`
- `tests/test_stage10_cold_dense.py`
- `tests/test_stage10_tradeoff.py`
- `tests/test_rcs114_unified.py`

## 跳过与保留

- 清单内因正式依赖或清单不一致而跳过的项目：无。
- 保留全部 Stage 5～8 仿真程序及完整 results/stage8/。
- 保留 experiments/run_stage10_cold20.py、plot_stage10_cold20.py、run_stage10_tradeoff.py、plot_stage10_tradeoff.py。
- 保留 results/stage10/cold_wide20_dl114_tol1e4/ 与 results/stage10/cold_wide14_tol1e4/ 全部文件。
- 保留 run_all_figures_continue.py 及其依赖，仍使用 14 点入口。
- 保留 tests/test_stage10a_warmstart.py：直接测试仍存在的核心算法接口，不依赖已移除诊断脚本。
- 保留清单外根目录 diagnose_stage10_local_convergence.py、test_stage10_cross_weight_warmstart.py，以及 Stage 10 根目录 CSV/NPZ。根目录局部收敛诊断脚本仅将 local_convergence_diagnosis 作为诊断输出路径，不属于正式程序读取依赖。
- 所有核心算法、所有 YAML、正式数据及论文图均未修改。保留文件基线共 224 项，最终 SHA-256 全部一致。

## 清理后验证

- Python 语法：51 个文件通过（内存 compile，无新字节码）。
- 模块导入：34 个根目录和保留实验模块通过。
- 正式入口：check_environment() 通过；总入口 Stage 10 simulate/plot 仍明确指向 14 点脚本。
- 测试收集：38 项正常收集。
- 非优化基础测试：tests/test_geometry.py、test_channels.py、test_metrics.py、test_smoke.py，共 5 passed。
- 未运行完整测试执行，因为其他测试包含 FP/RCG 优化调用；未重新运行任何数值优化。
- 20 点原始 CSV：4 方案 × 20 点，共 80 个原始收敛数据点通过校验。
- 已重新运行 plot_stage10_cold20.py --output-dir tmp/cleanup_completed_20261010/figures，成功生成 PNG/PDF/SVG，未覆盖正式图。
- 使用已有 D:/miniconda/envs/toma_isac/python.exe；未安装或修改环境。

## 审查记录

删除步骤曾被自动审批审查拒绝，理由是认为旧 results/stage10/figures 存在正式依赖。随后读取并导入正式模块，核对实际绝对路径与总入口 JOBS，证明正式目录分别是 stage8/figures、cold_wide14_tol1e4/figures、cold_wide20_dl114_tol1e4/figures；重新提交后执行成功，无遗留阻塞。证据见 figure_dependency_evidence.log。

## 审计文件

- archive_manifest.json：48 项目标、全部 2,881 个归档文件路径/大小/SHA-256、目录列表。归档根目录也保存了该清单。
- protected_before.json：224 个保留文件的校验基线。
- removed_targets.json：实际移除路径。
- final_integrity.json：最终完整性结果。
- validation.log、test_collection.log、tests.log、plot.log：检查记录。
