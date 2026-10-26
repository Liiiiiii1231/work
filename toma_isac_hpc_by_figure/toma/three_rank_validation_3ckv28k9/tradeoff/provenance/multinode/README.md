# 最终配置：Proposed 保留 FP-aware；MA 使用原始 MA + FP

当前压缩包虽然沿用 toma_MA_FPaware_no_extrap_o4_i5_all5_3nodes.zip 文件名，但已按用户要求覆盖更新：MA 不使用 FP-aware。请以本说明和实际调用代码为准。

- Proposed：保留原版 FP-aware。候选位置先执行最多两轮普通 FP，再进行上升判断；接受位置时同步继承候选资源。
- MA：直接调用 algorithms.run_ma_fp，保留原始 MA 外层位置搜索；候选位置不额外优化资源。MA 只调整容差。
- 固定位置方案：普通 FP，无位置搜索。
- 全部无外推加速。
- 外层：目标值绝对变化 <= 1e-4，连续两轮；内层：普通 FP 单步绝对变化 <= 1e-5。
- 正式上限内层1200、外层1500。之前400轮仅为对照测试预算。
- 五张图均为3节点、每节点56 CPU，每节点一个普通进程池，BLAS单线程。
- 天线数、场景、扫描范围、初始候选种子和Monte Carlo数量均不改变。基础16Tx/16Rx，8UAV，每缆4阵元。
- 筛选40任务；收敛90任务；02/03/04各2000任务；05为8000任务。独立任务数小于168时无法同时用满168个工作进程。

## 上传与提交

把包内 toma/ 的全部内容合并覆盖到 /home/lyq00008339/Lihuang_Li/toma/，不要形成 toma/toma/。新版本运行前停止仍在执行的旧代码作业。旧结果保留，旧 ma_fp_aware.py 即使残留，也不会被新 MA 调用，身份检查只记录实际参与的求解模块。

```bash
mkdir -p /home/lyq00008339/logs
cd /home/lyq00008339/Lihuang_Li/toma
unset HPC_SELECTION HPC_OUTPUT
sbatch 01_proposed_convergence/submit_convergence.sh
```

新筛选目录：results/final_initial_geometry_plainfp_o4_i5_mafp_final_s8005。01自动用三节点对 Proposed 和原始MA各20个初态重新评分，冻结各自获胜的 INITIAL 几何，再计算90个收敛任务。每个正式任务独立初始化资源，不采用筛选终态资源。

看到 FROZEN GEOMETRY READY 后，可以分别提交其余四图，不必等待01整张图结束：

```bash
sbatch 02_wsr_vs_p_dl_max/submit_p_dl_max.sh
sbatch 03_wsr_vs_rho_si/submit_rho_si.sh
sbatch 04_wsr_vs_system_load/submit_system_load.sh
sbatch 05_communication_sensing_tradeoff/submit_tradeoff.sh
```

已有MA FP-aware的筛选排名不作为这版MA的筛选结果，收敛图及各扫描图也应使用对应配置的新结果。只做本机缩小预算的三节点进程模拟，未实际提交集群。
