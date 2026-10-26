# 单场景：相邻候选筛选后重新优化

本脚本在集群直接读取run_33570的原始NPZ，不需要下载80组状态，不需要在VS Code上跑。只测试scene8001、20权重、4方案，共80个当前权重的优化任务。

## 本次规则

1. 按原始结果固定候选池，每个权重比较自身、直接左邻点和直接右邻点；端点只有一个邻点。
2. 按当前权重的WSR择优，不要求通信/感知单项都不降低；同分优先原解。
3. 将选中状态的完整几何、发射矩阵、上行幅度和接收合并器作为当前权重优化的起点。丢弃邻点原来的目标历史，用当前权重重新计算初始WSR。
4. 原求解器和内外层绝对容差1e-4保持不变，达到原停止条件才标记完成。到调用迭代上限仍未收敛时从同一状态继续，每次提交最多20次调用，可再次重交续跑。
5. 不顺序传播：某个点的新结果不再反过来修改本轮其他点的候选池。
6. 初态和终态资源/几何、配置哈希、状态哈希、真实WSR、轨迹单调性及停止差值均检查。目标WSR不能低于原点，但允许其中一类速率下降。

零上行幅度仍按原算法更新，未增加重新激活机制。本实验改变了起点协议，是单场景诊断，不能直接替换原100场景论文结果，也不构成全局最优证书。

## 上传：只需两个文件

放到集群目录：

/home/lyq00008339/Lihuang_Li/toma/05_communication_sensing_tradeoff/

- refine_neighbor_scene.py
- submit_neighbor_refine.sh

不覆盖run_tradeoff.py、共享算法文件或run_33570。其余共享代码使用之前已更新的7个公共Python与tradeoff_config.json（20权重）。

## 提交

平台作业脚本编辑器粘贴submit_neighbor_refine.sh的全部内容提交即可；或在集群终端：

```bash
sbatch /home/lyq00008339/Lihuang_Li/toma/05_communication_sensing_tradeoff/submit_neighbor_refine.sh
```

默认申请50 CPU、48GB、48小时；最多50个单点任务并行。source固定默认run_33570，输出为同级新的results/neighbor_refine_scene8001_作业ID，不改原结果。

如果原结果目录不是run_33570，只修改脚本SOURCE_RUN，或设置HPC_SOURCE为实际完整路径。

## 输出与续跑

全部80点均达到停止条件后才画图：

- figures/neighbor_refinement_comparison.png/pdf/svg：原始结果、筛选后、再优化后三栏对照。
- tables/selection.csv：每个点选用了哪个邻点。
- tables/original_per_scene.csv：单场景原始值。
- tables/refined_per_scene.csv：实际重新优化后的结果。
- refinement_tasks/：诊断专用状态和逐次调用记录。
- refinement_completion.json：完整性标记；有未完成点时不画部分曲线。

若超时或调用预算耗尽，保留整个新输出目录；重交时设置HPC_REFINE_OUTPUT为该目录，不改变配置/代码。完成点会校验后复用，未完成点从最后保存的状态继续。

```bash
export HPC_REFINE_OUTPUT=/home/lyq00008339/Lihuang_Li/toma/05_communication_sensing_tradeoff/results/neighbor_refine_scene8001_实际旧作业ID
sbatch /home/lyq00008339/Lihuang_Li/toma/05_communication_sensing_tradeoff/submit_neighbor_refine.sh
```

下载结果只需对照图及refined_per_scene.csv，文件数很少，不需要再次打包全部checkpoint。

## 本地验证范围

已用本地完整拥有的4个权重（0、0.001、0.015、0.2）×2方案（Proposed/MA），共8个真实检查点验证端到端流程：均真实重新启动优化并达到停止条件，输出图和CSV；随后collect-only重读状态校验通过。Bash语法通过。这个小网格的结果只用于验证流程，并不代表完整20权重规则已跑完。

完整20权重所需的部分选中NPZ目前本地没有，所以没有制作或声称存在完整20点重新优化后的图。集群端原始目录有这些文件，可直接运行两个上传脚本完成。
