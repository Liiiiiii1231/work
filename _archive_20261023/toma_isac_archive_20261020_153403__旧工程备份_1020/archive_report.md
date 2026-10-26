# toma_isac 历史实验归档报告

- 项目：`D:\Vs codeProjects\research work1\toma_isac`
- 归档：`D:\Vs codeProjects\research work1\toma_isac_archive_20261020_153403`
- 使用移动归档，归档目录在项目同级，保持原相对路径。
- 两次执行进程核查，未发现本机正在运行的项目 Python 正式仿真进程；未访问或提交远端 HPC 作业。

## 实际归档

共 118 个文件；37 个 experiments 历史脚本、3 个根目录脚本、9 个历史资源目录。

### experiments 脚本

- `experiments/check_proposed_near_field.py`
- `experiments/extend_stage8_rho_si_minus50.py`
- `experiments/planar_pm_initialization.py`
- `experiments/plot_convergence_early_main.py`
- `experiments/plot_proposed_convergence.py`
- `experiments/plot_stage10_cold20.py`
- `experiments/plot_stage10_ms20.py`
- `experiments/plot_stage10_tradeoff.py`
- `experiments/plot_stage8_results.py`
- `experiments/rerun_stage8_k5.py`
- `experiments/run_old_init_power_pilot.py`
- `experiments/run_stage10_cold20.py`
- `experiments/run_stage10_complete_fixed_abs.py`
- `experiments/run_stage10_continue_and_plot.py`
- `experiments/run_stage10_continue_b6_09.py`
- `experiments/run_stage10_fixed_ul_025mW.py`
- `experiments/run_stage10_fixed_ul_025mW_refined16.py`
- `experiments/run_stage10_ms20.py`
- `experiments/run_stage10_multistart20_probe.py`
- `experiments/run_stage10_refine_regular20.py`
- `experiments/run_stage10_refine_seed2042.py`
- `experiments/run_stage10_regular_planar_pm.py`
- `experiments/run_stage10_top10_abs.py`
- `experiments/run_stage10_tradeoff.py`
- `experiments/run_stage10_ul_init_compare.py`
- `experiments/run_stage10_ul_init_compare_representative.py`
- `experiments/run_stage10_warmstart_probe.py`
- `experiments/run_stage5_stress.py`
- `experiments/run_stage6_baselines.py`
- `experiments/run_stage7_multistart.py`
- `experiments/run_stage8_full_smoke.py`
- `experiments/run_stage8_load_sweep.py`
- `experiments/run_stage8_monte_carlo.py`
- `experiments/run_stage8_sweeps.py`
- `experiments/stage10_planar_pm_common.py`
- `experiments/stage10_seed2042_support.py`
- `experiments/trace_stage10_proposed_0125.py`

### 根目录脚本

- `run_all_figures_continue.py`
- `diagnose_stage10_local_convergence.py`
- `test_stage10_cross_weight_warmstart.py`

### 整体移动的资源目录

- `experiments/stage10_ul_compare_inputs/`
- `experiments/stage10_fixed_ul16_assets/`
- `experiments/stage10_fixed_ul_assets/`
- `experiments/stage10_regular20_source/`
- `experiments/old_init_pilot_assets/`
- `experiments/stage10_seed2042_assets/`
- `experiments/stage10_planar_pm_assets/`
- `experiments/stage10_ul_compare3_inputs/`
- `experiments/stage10_seed2042_diagnostic_refs/`

## experiments 最终保留

### Python 文件（12 个）

- `experiments/extend_regular25_scene8001_to_40dbm.py`
- `experiments/final_protocol.py`
- `experiments/ma_add4_common.py`
- `experiments/ma_add4_styles.py`
- `experiments/ma_add4_support.py`
- `experiments/paper_figure_styles.py`
- `experiments/paper_figures_unified.py`
- `experiments/run_regular025_multistart_pilot.py`
- `experiments/run_regular25_multistart_pilot.py`
- `experiments/run_stage10_ma_add4.py`
- `experiments/run_stage10_ma_transition4.py`
- `experiments/stage10_planar_pm_support.py`

### 资源目录（5 个，全部内容原路径保留）

- `experiments/ma_add4_inputs/`
- `experiments/ma_add4_solver_assets/`
- `experiments/ma_transition_inputs/`
- `experiments/paper_figures_assets/`
- `experiments/regular025_multistart_assets/`

## 追加保留依赖

`experiments/ma_add4_common.py:297` 的 main 函数通过 `Path(__file__).with_name('stage10_planar_pm_support.py')` 读取 SHA256，因此追加保留 `experiments/stage10_planar_pm_support.py`。这是保留源码中的运行时文件引用，不需要也没有修改 import、路径或校验逻辑。其余保留源码、资源及动态加载扫描未发现需要追加保留的依赖。原计划 38 个历史脚本因此实际归档 37 个。

## 清理前后检查

使用现有环境 `D:\miniconda\envs\toma_isac\python.exe`。以 `-B` 和 `PYTHONDONTWRITEBYTECODE=1` 禁止产生新缓存；未安装依赖。

| 检查 | 清理前 | 清理后 | 标准输出/错误输出 |
| --- | --- | --- | --- |
| unified | PASS (0) | PASS (0) | 字节一致 |
| ma_add4 | PASS (0) | PASS (0) | 字节一致 |
| ma_transition4 | PASS (0) | PASS (0) | 字节一致 |

实际检查命令（均在项目根目录执行）：

```text
D:\miniconda\envs\toma_isac\python.exe -B run_all_figures_unified.py --figures convergence p_dl_max rho_si system_load tradeoff --check-only --output results/cleanup_check_only_unused_20261020_153403
```

```text
D:\miniconda\envs\toma_isac\python.exe -B experiments/run_stage10_ma_add4.py --check-only
```

```text
D:\miniconda\envs\toma_isac\python.exe -B experiments/run_stage10_ma_transition4.py --check-only
```

临时输出参数 `results/cleanup_check_only_unused_20261020_153403` 在执行前不存在，执行后仍未创建。六次命令全部包含 `--check-only`，未调用优化器。完整 stdout/stderr 位于 `validation/`，细节见 `validation_before.json` 与 `validation_after.json`。

## 完整性与范围

- 118 个归档文件逐项核对原大小与 SHA256，全部一致，原路径已移走。
- 清理前后的 9962 个保留文件 SHA256 全部一致，无缺失、无修改、无额外新增项目文件。
- `configs/` 保留并验证 20 个文件。
- `results/` 保留并验证 9730 个文件。
- `paper/` 保留并验证 26 个文件。
- `hpc/` 保留并验证 3 个文件。
- 根目录其他 Python 文件、README_HPC.md、requirements.txt、UPLOAD_SHA256.json 均原字节保留；没有更新任何旧 SHA256 清单。
- 只清理 `experiments/__pycache__/` 下原有 4 个缓存文件；根目录缓存不存在，未递归清理 results、paper 或其他目录。
- 未发现移动引起的依赖缺失，无需回退恢复文件。
- 未新增测试脚本、未执行正式仿真、未重新打包项目。

## 已知版本差异

已按用户要求完整记录在 `known_version_differences.md`，仅报告，不修复。

## 清单与审计文件

- `archive_manifest.json`：每个移动文件的原路径、归档路径、大小、SHA256。
- `preserved_files_before.json` / `preserved_files_verification.json`：保留文件基线与最终复核。
- `dependency_audit.json`：保留模块 import / Python 文件引用及追加依赖。
- `audit_plan.json` / `process_check_before_move.json` / `cache_cleanup.json`：归档计划、进程核查和缓存记录。
- `archive_summary.json`：机器可读最终报告。
