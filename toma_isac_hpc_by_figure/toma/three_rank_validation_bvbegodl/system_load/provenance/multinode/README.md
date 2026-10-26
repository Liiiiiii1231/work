# 02–05：每张图 3 个节点，每节点 56 CPU

将补丁 ZIP 内 `toma/` 下的文件按相同相对路径覆盖到平台 `toma` 目录。

必须替换四个 `submit_*.sh`，并上传完整的 `multinode_support_fpaccel_o4_i5/` 新目录。原来的两个 FP 加速支持目录需要保留；本补丁不修改它们，也不修改 01 或五个共享求解器，因此当前 01 作业与生成的冻结文件仍然有效，**无需重新筛选**。

## 资源与调度

- 每图申请 3 节点、每节点 1 个 Slurm leader、每个 leader 56 CPU；总计最多 168 个计算进程。
- 每节点 48GB 内存，作业时限 48 小时，BLAS 单线程。
- `srun` 启动三个节点进程，每个节点内部再运行 56 个独立计算进程。不能仅把原单机脚本的 `-N` 改为 3，否则原 ProcessPoolExecutor 仍只在单节点运行。
- 用共享文件系统上的原子 mkdir 领取独立任务。每个节点先领取自己的任务，再接手其他节点尚未领取的任务；不重复计算，不改变场景或权重初始化。
- 只有 rank 0 在所有节点结束后验证并汇总全部结果、出图。完成记录不足或任意任务出错时拒绝计算部分均值；单项达到迭代上限仍按原协议保留未收敛标记，不自动续算。
- `metadata/multinode_summary.json` 记录每节点主机名、分配 CPU、任务数量；`distributed/claims`、`receipts` 保存唯一领取和完成记录，便于确认三个节点实际参与。

需要平台的 `/home/.../toma` 及结果目录在三个节点上共享可见，且三个节点能使用同一个 Conda 环境。这与通过共享目录读取原工程的部署方式一致。调度参数参照 [Slurm srun](https://slurm.schedmd.com/srun.html) 和 [sbatch](https://slurm.schedmd.com/sbatch.html) 官方说明。

## 提交

先等待现有 01 输出 `FROZEN GEOMETRY READY`，然后分别提交：

```bash
cd /home/lyq00008339/Lihuang_Li/toma
sbatch 02_wsr_vs_p_dl_max/submit_p_dl_max.sh
sbatch 03_wsr_vs_rho_si/submit_rho_si.sh
sbatch 04_wsr_vs_system_load/submit_system_load.sh
sbatch 05_communication_sensing_tradeoff/submit_tradeoff.sh
```

四张图各自占 3 个节点，若同时运行共 12 节点、672 CPU。输出路径仍在各图的 `results/final_3nodes_fpaccel_o4_i5_作业ID` 下。不要将 HPC_OUTPUT 指向旧结果；不复用或续算已有输出。

天线、几何、用户/目标、MC 数、权重点、功率、内外容差、FP 加速及冻结身份校验均不变。02/03/04 各 2000 项，05 共 8000 项。`--local-test` 仅供本地多进程调度验证，不能用于论文正式数据。实际平台提交只能通过上述三节点 srun 入口。
