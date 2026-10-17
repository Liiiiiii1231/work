# 最终冻结仿真上传与运行

只需上传 `toma_isac_hpc_mc100.zip`，解压后进入 `toma_isac_hpc_mc100/`。包内全部保留：根目录核心 Python 文件、`configs/`、`experiments/`、`hpc/`、依赖及校验清单。不要上传旧 results、paper、tests、临时文件、虚拟环境和历史项目压缩包。

## 冻结参数

- 五图均为 100 场景，种子 8001–8100，同场景跨方案和扫描参数共用用户位置。沿用原有位置模型：用户/目标基准坐标加 x/y 独立均匀扰动 ±0.5 m，z 不变，纯 LoS。
- ToMA 第 0 起点：x=0 平面均匀排布，端点角度 45°、135°、225°、315°；MA/UPA 使用已有规则阵列。
- 下行 MRT 初始化，感知波束按目标方向初始化，上行接收匹配滤波，感知接收主特征向量；之后运行原有 FP 资源优化。
- 所有图和用户的初始上行功率为 25 mW（0.025 W，幅度 sqrt(0.025)），功率上限保持 100 mW。
- 内外层用相邻目标值绝对差 ≤1e-4。联合位置优化连续两步满足条件；固定位置方案检查内层停止。禁用窗口平台早停。单次调用最多内层 1200 / 外层 1500 次，最多调用 20 次；未收敛报错，可原命令续跑。
- 收敛/功率/SI/负载图：Proposed 5 起点、MA 3 起点、固定 ToMA/UPA 各 1 起点。第 0 起点规则排布，其他起点沿用可行几何扰动；扰动和优化不限制在初始平面。每场景全部起点收敛后选最佳 WSR，再平均 100 场景。
- 权重图：每场景每方案只用同一个规则初始天线位置，资源随机数种子统一 2042、跨权重不变。用户位置仍随场景变化；每个权重独立初始化，无跨权重 warm start。目标 w*(R_DL+R_UL)/10+(1-w)*R_S/5。全部重算，不复用旧 0.25 mW 结果。
- 四方案、权重及扫描点沿用统一配置。权重图显示全部平均点，不平滑、不做非支配筛选。
- 收敛图每场景先选最佳起点再平均轨迹；已收敛场景在其他场景继续时保持终值，CSV 保留真实逐场景轨迹。

## 集群运行

采用所提供 slurm-hpc-experiment-runner 模板的 CPU-only、单层数组并行、共享存储保存和线程限额。截图没有实际环境路径，脚本要求指定目标 Python。

```bash
cd /你的上传路径/toma_isac_hpc_mc100
export HPC_PYTHON=/你的conda路径/envs/你的环境/bin/python
"$HPC_PYTHON" -c 'import numpy, scipy, yaml, matplotlib; print("dependencies OK")'
sinfo -o '%P|%a|%l|%D|%N'
bash hpc/submit.sh
```

缺依赖时在自己的环境中按 requirements.txt 安装。核心求解器为 NumPy/SciPy 实现，无需 CVXPY。脚本先轻量预检，再交 Slurm 正式计算。

默认 cpu 分区，10 个数组任务（编号 0–9）、最多 10 并发；每任务 1 CPU、4 GB、72 小时，共最多使用 10 CPU / 40 GB。每个 CPU 顺序处理 10 个蒙特卡洛场景：任务 0 处理场景索引 0–9，任务 1 处理 10–19，依此类推，任务 9 处理 90–99。每场景运行全部五图的 229 次起点优化，总共 22900 次。每个进程的 BLAS 线程数固定为 1。实际并发取决于 Slurm 资源调度，数组任务可分布在不同节点。实际时间/内存需集群试跑确认，可改 hpc/run_array.sh 的资源参数。日志目录在 sbatch 前创建；禁止登录节点正式计算。

提交会打印作业 ID、唯一结果目录，并创建依赖全部数组任务成功的汇总作业。汇总验证每个起点、配置、文件哈希和绝对收敛，缺场景不生成该图平均结果。

```bash
squeue -u "$USER"
sacct -j 作业ID --format=JobID,JobName,State,Elapsed,ExitCode,MaxRSS
```

部分任务超时/未收敛时，使用原目录补跑失败的批次编号（示例 0、1，对应场景索引 0–19）；批次内已完成的场景会校验并复用断点。单个场景失败后会继续该批次剩余场景，最终返回失败状态，避免汇总未完成结果。

```bash
export HPC_OUTPUT=results/原始结果目录
sbatch --array=0,1%10 --export=ALL hpc/run_array.sh
# 补跑成功后汇总
sbatch --export=ALL hpc/collect.sh
```

旧 afterok 汇总如因原数组失败而一直等待，可取消该汇总作业。新实验用新目录；代码或配置变更不得混用旧 checkpoint。仅在同参数、场景、权重处续跑，保存初始状态及每次调用的状态。

结果目录包含 figures/（PNG/PDF/SVG）、tables/（起点、逐场景和均值）、candidate_*/tasks/（状态、轨迹、有效配置）、metadata/、status.json、completion.json。确认 Slurm 成功且 complete=true 后用于论文。

本地预检：`python run_all_figures_unified.py --check-only`。入口默认五图及 100 场景。`--n-mc` 和 `--figures` 只供小规模验证，正式脚本固定 100 场景。本交付仅本地验证和打包，尚未连接集群、提交作业或获得 100 场景结果。
