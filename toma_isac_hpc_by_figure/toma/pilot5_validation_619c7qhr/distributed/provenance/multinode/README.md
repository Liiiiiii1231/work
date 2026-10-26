# 权重图基线预算小规模测试：5节点 × 每节点56 CPU

这是作业33885权重图的独立诊断测试，不替换正式01—05代码，不修改原结果，不重新筛选位置。

## 测试设置

- 10个场景：8006、8010、8011、8013、8015、8028、8035、8042、8092、8099。
- 选取2个较容易、2个典型、6个困难场景；详细规则和旧结果尾部增量见 selected_scenes.json。是针对性诊断样本，不能代替无偏的100场景论文均值。
- 通信权重：0、0.1、0.2、0.6、1。
- MA + FP：保留原始MA位置搜索，内层上限1200，外层上限3000。
- Fixed-ToMA + FP、Fixed UPA + FP：内层上限5000，位置固定。
- 容差不变：内层普通FP绝对1e-5；MA外层绝对1e-4、连续两轮。无外推、无MA FP-aware，无平台停止或自动续算。
- 合计150个独立任务。申请5节点共280核，每节点56CPU；任务少于280，因此最多并发150个任务，不能同时用满280核。任务首先分配到全部5个节点，再允许节点间抢占剩余任务。
- 使用原8000任务结果包中的精确 initial.npz 和 effective_config，从原始位置及资源重新跑，不从旧终态续算。inputs/和runtime/必须随包上传。无需上传原大型结果包，也不依赖服务器上的冻结筛选目录。
- runtime/中的原版算法/模型代码来自作业33885的provenance；与输入记录逐项校验配置和初态SHA256。天线、信道、场景、权重和初始资源不改。

## 上传与提交

上传压缩包内整个 tradeoff_budget_pilot_5nodes/ 文件夹到：

```text
/home/lyq00008339/Lihuang_Li/toma/tradeoff_budget_pilot_5nodes/
```

不要覆盖现有正式仿真支持文件夹。提交命令：

```bash
mkdir -p /home/lyq00008339/logs
cd /home/lyq00008339/Lihuang_Li/toma/tradeoff_budget_pilot_5nodes
sbatch submit_pilot.sh
```

默认使用 /home/lyq00008339/miniconda3/etc/profile.d/conda.sh 和 JM_CVX 环境。可通过 CONDA_SH、TARGET_CONDA_ENV、HPC_PYTHON 调整。不要手动用 python 启动正式150任务，五节点运行应通过 sbatch/srun。

如只检查输入、配置和预算，不启动优化：

```bash
python run_pilot.py --output results/check_only_unused --check-only
```

正式输出在 results/pilot_budget5_作业ID/。不读取通用 HPC_OUTPUT 或 HPC_SELECTION；如需换目录，请用 PILOT_OUTPUT 指向共享存储中的新空目录。

## 下载哪些结果

全部任务运行结束后，会自动产生：

```text
results/pilot_budget5_作业ID_bundle.zip
```

下载此ZIP即可用于判断预算是否够用。里面包括：

- tables/pilot_per_task.csv：每个任务的新/旧收敛状态、轮数、通信/感知速率、目标值变化、总FP更新次数、尾部增量。
- tables/pilot_summary.csv：每个权重/方案在相同选定场景上的汇总。
- figures/pilot_budget_comparison.png / .pdf：同一小样本下原预算与更大预算的结果对比。
- completion.json：任务是否全部结束、实际收敛数量；“全部结束”不等于“全部收敛”。
- tasks/：每个任务的完整新收敛历史和终态；metadata/和distributed/：节点、任务归属及代码身份信息。

original_prefix_max_difference 对比新旧轨迹共同前缀；不同CPU/数值库可能产生数值路径差异，应结合该字段判断可重复性。runtime_s为本次150任务并行运行下的时间，不宜直接与旧8000任务混合负载下的runtime_s当作严格加速比。

## 如何决定下一步

先看每个方案的 new_converged/new_capped，以及未收敛任务的 last_increment、last100_gain；再看同一任务的 delta_comm、delta_sensing 和 relative_wsr_change。3000/5000是试测预算，不保证全部任务足够。此测试不自动放宽容差，也不自动重跑正式100场景。

validation_report.json 只确认本机缩小预算的5节点进程模拟；实际HPC五个物理节点和正式预算仍需平台运行验证。
