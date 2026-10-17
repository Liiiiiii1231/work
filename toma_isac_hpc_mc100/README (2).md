# 9 文件集群仿真包（提交脚本单独提供）

将本目录的全部 9 个文件上传到集群的同一个目录。cluster.sh 单独提供，不在上传包内；运行前将其内容复制到平台脚本编辑器，或在工程同目录新建 cluster.sh 并粘贴保存。没有额外配置目录或代码目录需要上传，也不需要为每张图复制一份模型。

| 文件 | 职责 |
| --- | --- |
| system_model.py | 系统参数、状态类型、几何模型、约束检查、随机用户场景 |
| channel_model.py | 通信/感知/自干扰信道、SINR/速率指标、MRT 等资源初始化 |
| inner_solver.py | 内层 FP 资源优化 |
| outer_solver.py | 外层位置优化 |
| algorithms.py | 联合 AO 调度、固定 ToMA/UPA 基线、MA 基线 |
| figure_engine.py | 五图任务生成、多起点、断点校验、指标汇总和绘图 |
| run_figures.py | 仿真命令行入口 |
| simulation_config.json | 统一协议、系统参数及 3–7 用户的全部负载配置 |
| README.md | 本说明及依赖 |

## 环境和提交

Python 环境需要 numpy、scipy、matplotlib。算法为 NumPy/SciPy 实现，不调用 CVXPY/MOSEK，无需 MOSEK 许可证。配置使用 JSON，不需要 PyYAML/yaml。

单独提供的 cluster.sh 采用单个 Slurm 作业的模板：cpu 分区、1 节点、10 CPU、40 GB、最长 72 小时。作业内启动 10 组后台任务，每组顺序执行 10 个场景。BLAS 固定单线程；全部任务完成后在同一作业内汇总。没有作业数组，也没有二次 sbatch 提交。运行时间仍需在目标集群实测。

用户确认的代码绝对路径为 /home/lyq00008339/Lihuang_Li/toma。脚本直接固定 PROJECT_ROOT=/home/lyq00008339/Lihuang_Li 和 CODE_DIR=/home/lyq00008339/Lihuang_Li/toma，不读取提交目录，也不自动查找代码路径。

环境默认沿用用户提供的模板：

```bash
TARGET_CONDA_ENV=JM_CVX
CONDA_SH=/home/lyq00008339/miniconda3/etc/profile.d/conda.sh
```

JM_CVX 仅为环境名称，并非要求使用 CVX。环境和 Conda 初始化路径尚未连接集群核实。脚本可以保存在平台作业编辑器或 ~/jobsh/，运行时会进入固定的 CODE_DIR。

将单独提供的脚本保存为 cluster.sh，然后提交全部五图：

```bash
sbatch cluster.sh
```

仅提交一张图时选择其中一条：

```bash
sbatch cluster.sh convergence
sbatch cluster.sh p_dl_max
sbatch cluster.sh rho_si
sbatch cluster.sh system_load
sbatch cluster.sh tradeoff
```

若使用平台作业页面，粘贴 cluster.sh 全部内容并提交；默认全部五图。单图可填写脚本参数，或把脚本 FIGURE 默认值从 all 改成对应图名。资源设为 1 节点、每任务 10 CPU，不设置数组。若平台资源选项覆盖 #SBATCH，以平台选项为准。

只进行轻量预检：在已激活的 Python 环境和 toma 目录中执行 python run_figures.py --check-only。正式计算必须通过 sbatch 或平台提交，脚本会检查已分配 10 CPU，避免在登录节点直接运行。

各图共用同一份系统、信道和算法代码。每个作业申请 10 CPU；五张图分别同时提交会申请 50 CPU。若总共只使用 10 CPU，提交默认全部五图，或等一张图完成后再提交下一张。

## 断点与汇总

结果目录默认 results/图名_mc100_作业ID，主日志为 %j.out / %j.err；每组详细日志在结果目录 logs/batch_0.out 至 batch_9.out 及对应 err 文件。

单场景失败后继续该组剩余场景。任一组未完成则作业返回失败，不汇总不完整结果。续跑指定原目录和原图名（示例功率图）：

```bash
export HPC_OUTPUT=results/p_dl_max_mc100_原作业ID
sbatch cluster.sh p_dl_max
```

会重新遍历 100 场景，校验并复用已完成断点，只继续未完成的优化。如需重新汇总，也使用原目录和原图名再次提交；脚本会复用已完成断点并在最后汇总。

运行生成 figures/、tables/、candidate_*/tasks/、metadata/ 等子目录，不属于需要上传的 9 个输入文件。所有起点和 100 场景收敛后才生成该图平均结果。代码或配置变化后使用新结果目录。本合并版不与旧多文件包 checkpoint 混用。调度方式变更而 Python 和仿真配置未变时，可以继续原合并版的兼容断点。

## 冻结协议

- 五图均为 100 场景，场景种子 8001–8100；同场景跨方案和参数复用用户位置。用户/目标位置为原基准点加 x/y 均匀扰动 ±0.5 m，z 不变，纯 LoS。
- 初始上行功率 25 mW，上限 100 mW。内外层绝对差 ≤1e-4，位置联合优化连续两步满足条件；关闭窗口平台早停。
- ToMA 第 0 起点为 x=0 平面、端点角度 45/135/225/315 度；MA/UPA 为规则阵列。下行 MRT、感知方向波束、上行匹配滤波和感知接收主特征向量初始化，然后进行原有 FP 优化。
- 非权重图采用 Proposed 5 起点、MA 3 起点、固定 ToMA/UPA 各 1 起点；规则第 0 起点之外使用原可行扰动，扰动和优化不限制在初始平面。每场景取收敛起点中最佳 WSR，要求全部起点完成后再平均。
- 权重图跨权重共用规则初始天线位置、资源随机数种子 2042，每场景仍有不同用户位置，无跨权重 warm start，不复用旧 0.25 mW 结果。
- 权重目标 w*(R_DL+R_UL)/10+(1-w)*R_S/5；各图扫描点与此前冻结版本一致。收敛图平均轨迹对提前停止场景保留其终值，并保存真实逐场景轨迹。
- 所有五图总计 22900 次起点优化。尚未在集群执行完整 100 场景仿真。

本地预检：python run_figures.py --check-only。单图预检：python run_figures.py --figures p_dl_max --check-only。所有图均引用上述共用模块。
