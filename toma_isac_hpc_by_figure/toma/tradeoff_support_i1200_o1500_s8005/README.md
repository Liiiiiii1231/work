# 当前最终仿真：参考场景8005，排除8001

当前入口：01到04使用final_support_i1200_o1500_s8005；05使用tradeoff_support_i1200_o1500_s8005。
这两个新支持目录必须完整上传到平台toma下面，包括PY、JSON、CSV、MD等文件。
替换01..05目录内的入口PY、提交SH、配置JSON及说明MD，不需要上传本地results，也不要删除平台旧results。
五个共享模块algorithms.py、channel_model.py、inner_solver.py、outer_solver.py、system_model.py未修改，不需替换。

## 运行顺序
旧33769仍在筛选8001，不会自动切换到8005；如决定改用此版本，取消旧作业后重新提交01。旧结果无需删除。
cd /home/lyq00008339/Lihuang_Li/toma/01_proposed_convergence
sbatch submit_convergence.sh
01在参考场景8005、30dBm、原基准权重下，对Proposed和MA各跑20个起点，共40项独立筛选。两方案各选各的获胜初始几何。
冻结的是获胜INITIAL几何，不是最终优化几何或资源；后续每项独立重建资源，不使用邻近参数或权重点热启动。
新冻结文件：/home/lyq00008339/Lihuang_Li/toma/results/final_initial_geometry_i1200_o1500_s8005/frozen_initial_geometry.json
看到FROZEN GEOMETRY READY后，即可分别提交02、03、04、05，不必等待01全部结束。
其他提交脚本依次是submit_p_dl_max.sh、submit_rho_si.sh、submit_system_load.sh、submit_tradeoff.sh。
每张图独立申请1节点56核、48GB、48小时，动态领取任务；账户配额和尾部任务数会限制实际并发。
不要使用旧HPC_SELECTION变量指向旧冻结文件；新脚本默认使用上述新路径。

## 样本和模型
收敛图继续使用原先冻结的30个历史较快场景，名单本身不含8001；3个功率点，共90项。
功率图、干扰图、负载图和权重图统一使用场景8002..8101，共100次MC。
前三张参数图各2000项；权重图20权重乘100场景乘4方案，共8000项。
20权重：0, 0.001, 0.005, 0.015, 0.03, 0.05, 0.075, 0.1, 0.125, 0.16, 0.18, 0.2, 0.24, 0.28, 0.32, 0.36, 0.42, 0.6, 0.8, 1。
内层1200、外层1500，绝对容差1e-4，不自动续算未收敛结果。Proposed与MA均使用FP-aware；固定阵列布局保持原定义。
主无人机20米高；地面全方位圆环水平半径sqrt(1200)±10米按面积采样；16发16收，收发各4架子无人机、各4阵元。
排除8001是根据已观察耗时作出的选择，因此样本经过筛选；8005在不同参考目标下不保证更快。协议和completion.json记录此规则，论文需说明。

## 结果
各图输出在自身results/final_作业ID或final_tradeoff_作业ID，保存CSV、PNG/PDF/SVG、NPZ、日志、源码、冻结文件。
完成后汇总all_data.json并打包；未收敛结果保留标记。四图合并仅匹配同一新冻结文件，权重图单独打包。
旧支持目录为正在运行的旧作业保留；新脚本不再使用它们，避免运行时加载到被修改的代码。
