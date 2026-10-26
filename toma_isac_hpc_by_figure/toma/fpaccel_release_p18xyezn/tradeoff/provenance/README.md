# 当前论文仿真版本：外层 1e-4，内层 1e-5，资源 FP 外推加速

01–04 使用 `final_support_fpaccel_o4_i5_s8005`；05 使用 `tradeoff_support_fpaccel_o4_i5_s8005`。旧支持目录及五个根目录共享求解模块保持不变，旧作业可继续使用旧目录。只上传新版本、01–05 的入口/脚本/配置与本说明，不覆盖平台旧结果。

## 求解协议

- Proposed、MA 的外层：真实联合 WSR **绝对增量 ≤ 1e-4，连续两轮**；最多 1500 轮。
- 四方案主资源求解统一使用带功率投影和真实 WSR 单调验收的 FP 平方外推。内层：接受的外推块绝对目标增量 ≤ 1e-5，连续两块；每次主内层最多 **1200 次实际普通 FP 更新**，包含被拒试探。外推系数上限 64。
- 一个外推块通常执行三次普通 FP；回溯时可能更多。日志分别保存外推块数和实际 FP 次数，不将二者混用。
- Proposed 和 MA 的位置候选仍使用**最多两次普通 FP**；不对位置候选使用外推加速。接受的位置和资源同步更新。
- 不启用窗口平台早停，不续算预算截止任务，不跨参数/权重热启动；达到上限不等于收敛。固定阵列方案以主内层实际 FP 次数统计迭代数，保存的资源轨迹横轴为接受的外推块。
- `accelerated_inner_diagnostics.json` 保存每次主内层的真实 FP 次数、外推块、目标增量及是否达到内层容差；Proposed 另外保存位置诊断和最终 RCG 记忆。

主资源加速器已独立封装在两个支持目录内。论文算法描述应包含平方外推、功率投影、单调验收/回退和两块停止规则，不能继续表述为未加速的逐步 FP。

## 新冻结目录与提交顺序

算法和精度已改变，旧冻结文件身份校验不会放宽。新选择目录：

`/home/lyq00008339/Lihuang_Li/toma/results/final_initial_geometry_fpaccel_o4_i5_s8005`

先提交 01，在场景 8005、30 dBm、基准权重下，分别对 Proposed 与 MA 各筛选 20 个起点。按各自最终 WSR 选取获胜起点，冻结其**初始几何**；后续任务从冻结几何独立初始化资源。不要将旧文件改名或手工改身份来跳过筛选。

```bash
cd /home/lyq00008339/Lihuang_Li/toma
sbatch 01_proposed_convergence/submit_convergence.sh
# 日志出现 FROZEN GEOMETRY READY 后，分别提交其余四张图：
sbatch 02_wsr_vs_p_dl_max/submit_p_dl_max.sh
sbatch 03_wsr_vs_rho_si/submit_rho_si.sh
sbatch 04_wsr_vs_system_load/submit_system_load.sh
sbatch 05_communication_sensing_tradeoff/submit_tradeoff.sh
# 或在冻结文件就绪后使用：bash submit_all_other_figures56.sh
```

若环境中残留 `HPC_SELECTION`，请取消该变量或指向上面的新目录。新脚本支持 `TOMA_ROOT`、`CONDA_SH`、`TARGET_CONDA_ENV` 覆盖平台路径/环境。每图独立申请 1 节点、56 核、48GB、48 小时，进程内 BLAS 单线程；新输出目录带 `fpaccel_o4_i5` 标识。

## 样本、模型与结果

几何、信道和目标权重保持既定协议：16 发 16 收，收发各 4 架子无人机，每根缆绳 4 阵元；主无人机高 20 米；地面全方位圆环半径 sqrt(1200)±10 米，按面积均匀抽样。

收敛图使用历史选出的 30 个场景、20/30/40 dBm，共 90 项；其余参数图/权重图使用 8002–8101 共 100 场景。权重仍为既定 20 点，权重图 8000 项。排除 8001 与历史场景筛选规则仍如实记录，不能把该筛选解释为加速器效果。

每图保存 CSV、PNG/PDF/SVG、初末 NPZ、源码、配置身份及冻结文件。`complete` 表示预算任务完成，`all_converged` 才表示全部满足停止条件。四图合包只匹配同一个新冻结文件；旧版本结果不会混入。

上传文件及校验值见 `UPLOAD_FPACCEL_O4_I5.json`；本地验证结果见新支持目录的 `validation_report.json`。本地验证不代表已完成正式 MC，也未向 Slurm 提交作业。
