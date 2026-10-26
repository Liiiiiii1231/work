# 原版 FP-aware：外 1e-4 / 内 1e-5

本版本移除额外的 FP 外推加速。Proposed 保留原版 FP-aware：候选位置最多进行两轮普通 FP，接受位置时同时继承该候选的资源。MA 保留原有 FP-aware 位置搜索；固定位置方案使用普通 FP。

外层：目标值绝对变化 <= 1e-4，连续两轮；内层：普通 FP 单步目标值绝对变化 <= 1e-5。内层上限 1200，外层上限 1500；400 轮仅是之前配对复测的对照预算，不是正式全图上限。无自动续算或平台停止。

天线数、用户/目标数、功率、信道、场景种子、扫描范围和 Monte Carlo 数量均沿用原版。基础阵列 16 Tx / 16 Rx，8 UAV、每缆4阵元。01 为30场景 x 3功率=90任务；02/03/04 各2000任务；05 为8000任务。

## 上传

将压缩包内 toma/ 的内容合并覆盖到 /home/lyq00008339/Lihuang_Li/toma/。保留旧结果及旧支持目录；新入口只调用 plainfp 目录。不要再上传旧 fpaccel 补丁覆盖新入口。包内五个共享算法/模型模块与旧原版完全一致，附带它们以保证完整性。

## 重新筛选与运行

本版使用新的筛选目录 results/final_initial_geometry_plainfp_o4_i5_s8005，绝不自动采用旧的加速版筛选结果。新任务前请清除曾在终端设置的 HPC_SELECTION 和 HPC_OUTPUT 环境变量，避免它们覆盖新默认目录。

```bash
mkdir -p /home/lyq00008339/logs
cd /home/lyq00008339/Lihuang_Li/toma
unset HPC_SELECTION HPC_OUTPUT
sbatch 01_proposed_convergence/submit_convergence.sh
```

01 使用3节点、每节点56 CPU，筛选和收敛两个阶段都使用三节点任务队列，自动在场景8005、30dBm下分别测试 Proposed/MA 的20个候选初始几何（共40任务），冻结各自获胜的 INITIAL 几何，再运行90个收敛任务。只冻结初始几何，不继承筛选终态资源。候选种子不变，用原版求解器重新评分，获胜位置可能改变。

看到 01 日志 FROZEN GEOMETRY READY 后，即可分别提交下列四图，不必等待01整张收敛图完成：

```bash
sbatch 02_wsr_vs_p_dl_max/submit_p_dl_max.sh
sbatch 03_wsr_vs_rho_si/submit_rho_si.sh
sbatch 04_wsr_vs_system_load/submit_system_load.sh
sbatch 05_communication_sensing_tradeoff/submit_tradeoff.sh
```

五张图均为每图3节点、每节点56 CPU，共168并行进程槽位；其余四图同时运行共12节点672 CPU；五图同时运行共15节点840 CPU。BLAS单线程，结果放共享 /home；srun 在每节点启动一个56进程池。日志位于 /home/lyq00008339/logs/作业名_作业ID.out/err。

旧加速版收敛图及其他图不能作为本版的算法结果。若坚持“同一协议下20候选择优”的筛选规则，需要重新筛选一次。已有位置本身可以作为任意固定初始几何，但不能宣称它是本版重新筛选的优胜者；本包采用重新筛选流程。

只做了本机缩小预算的多进程验证，没有实际提交HPC作业；完整1500轮/100场景结果须在集群上运行。
