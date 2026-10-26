# 最终四图替换包（不含权重图）

## 上传和替换

ZIP中的01_proposed_convergence、02_wsr_vs_p_dl_max、03_wsr_vs_rho_si、04_wsr_vs_system_load四个目录，合并到平台原toma下的同名目录，只替换各自的运行入口和提交脚本。
新增final_support_20261023目录必须完整上传到toma下，包含PY、JSON、CSV，不能只上传Python文件。
父目录algorithms.py、channel_model.py、inner_solver.py、outer_solver.py、system_model.py均不需要替换。
05_communication_sensing_tradeoff和已有results均不修改。建议先保留平台旧入口脚本备份。
本地旧入口已经备份至other_figures_entry_backup_20261023。

## 按四次提交运行

1. 首先提交01_proposed_convergence/submit_convergence.sh。
2. 01先在参考场景8001、DL功率30dBm、rho_SI=1e-10、5DL+5UL+5目标的基准模型中，对Proposed和MA分别运行20个几何起点，共40项优化。
3. 各方案独立按终态目标最大选胜出起点；若分数相同，选编号较小者。筛选中达到预算但未收敛的状态会保留标记，并按其实际目标参加比较。
4. 冻结胜出起点的初始几何，绝不把最终优化位置或资源当作后续初态。每个正式任务从冻结几何独立重建MRT/目标导向等资源，上行每用户25mW。
5. 日志出现FROZEN GEOMETRY READY后，02、03、04分别提交，不必等待01整张图画完。

冻结文件：
    /home/lyq00008339/Lihuang_Li/toma/results/final_initial_geometry_20261023/frozen_initial_geometry.json

命令：
    cd /home/lyq00008339/Lihuang_Li/toma/01_proposed_convergence
    sbatch submit_convergence.sh

等到FROZEN GEOMETRY READY后，再分别执行：
    cd /home/lyq00008339/Lihuang_Li/toma/02_wsr_vs_p_dl_max
    sbatch submit_p_dl_max.sh
    cd /home/lyq00008339/Lihuang_Li/toma/03_wsr_vs_rho_si
    sbatch submit_rho_si.sh
    cd /home/lyq00008339/Lihuang_Li/toma/04_wsr_vs_system_load
    sbatch submit_system_load.sh

网页提交也可直接粘贴对应.sh文件完整内容；须先把目录文件上传齐全。
四个作业分别申请1节点、56核、48GB、48小时，每进程BLAS单线程。
每个作业内部动态领取独立任务，不能保证最后剩余少量任务时仍用满56核。
是否允许四个56核作业同时运行取决于账户配额和调度器；单节点配额时其余作业会排队，不会由程序强占服务器其他节点。

## 同步模型和算法

- 收发各4架子无人机，每根1米缆绳4个等距阵元，即16发16收。
- 主无人机离地20米，用户和目标在地面圆环内按面积均匀独立采样，水平半径sqrt(1200)±10米，即24.64至44.64米；方位0至360度。
- 同场景、同系统负载下，跨方案和参数的对象坐标一致。系统负载图会按用户/目标数生成相应数量对象。
- Proposed、MA都使用候选最多2轮FP的FP-aware位置验收；MA修改放在新支持目录的ma_fp_aware.py，运行时适配原算法，不覆盖共享算法。
- Fixed-ToMA和Fixed UPA维持原固定基准布局，只有阵元数与场景同步更新；不会使用优化后的布局冒充固定基线。
- 发射总功率、噪声、RCS、感知波束列数规则和忽略UL到DL交叉干扰等其余建模假设保持原配置。
- 内层最多1500轮、外层最多2000轮，绝对容差1e-4。每项任务只调用一次，不自动续算。已有记录只校验复用，不继续其未收敛优化。

## 四图样本与工作量

01：30个历史筛选场景，20/30/40dBm三个功率点，仅Proposed，共90项；另承担一次40项起点筛选。
02：8001至8100共100场景，功率20/25/30/35/40dBm，四方案，共2000项。
03：8001至8100共100场景，残余SI横轴保持-50/-40/-30/-20/-10dB，四方案，共2000项。
04：8001至8100共100场景，K=J=L为3/4/5/6/7，四方案，共2000项。
总计6130项独立优化调用，其中起点筛选40项只做一次。此工作量远大于前面的140项试跑，不能沿用其总耗时估计。

收敛图冻结的30个场景：
8003,8027,8060,8072,8016,8080,8032,8042,8088,8052,8091,8097,8065,8043,8026,8029,8095,8075,8024,8006,8041,8057,8045,8059,8038,8055,8030,8094,8085,8082。
筛选按旧100场景每功率最高WSR起点的迭代数，先按三个功率的最大轮数排序，再按平均轮数、场景编号排序，取前30。
图中明确标记historically selected scenes；这是经过快收敛筛选的样本，不是随机选出的无偏30次MC。旧模型与新模型不同，历史快不保证新配置仍快。

## 输出与自动打包

每张图输出到自己的results/final_作业ID，自动生成CSV、PNG/PDF/SVG、NPZ完整状态、日志、all_data.json及同级final_作业ID_bundle.zip。
01包还包含完整40项筛选记录；每张图都带冻结几何文件和筛选对比表。
程序记录真实converged标记；complete仅表示全部预算任务已结束，不代表全部收敛。未收敛终点会进入预算结果图，平均值不剔除慢场景，也不对缺失场景做部分平均。
等四张图均完成导出后，最后结束的作业自动将四图结果合并到：
    toma/results/final_four_figures_冻结标识.zip
并输出同名JSON总表。不会把不同冻结起点的实验混在一起。
如需手动重打总包：
    python /home/lyq00008339/Lihuang_Li/toma/final_support_20261023/collect_four.py --selection-dir /home/lyq00008339/Lihuang_Li/toma/results/final_initial_geometry_20261023

父目录模型和求解器有哈希检查；若服务器文件不匹配，不绕过校验，先核对版本。
改变源码/配置后不能往原结果或冻结目录续跑，应另用HPC_OUTPUT和HPC_SELECTION指定新的目录。
程序正常或Python报错返回时会打包已保存结果；平台强制结束可能无法触发最终导出。

## 验证范围

已验证每方案20个不同可行起点、冻结的是胜出初始几何、跨场景/参数复用几何且重新初始化资源、四图完整任务计划。
使用截短预算执行40项筛选和38项正式流程测试，四图均成功绘制和导出。
另验证四张图单包及最后自动生成的总包完整性。这些是流程测试，不是正式6130项实验结果；未代用户提交平台。
