# 最终权重图：1200/1500，复用收敛图筛选的初始几何

将包内05_communication_sensing_tradeoff合并到平台toma的同名目录，并把tradeoff_support_i1200_o1500整个目录放在toma下。05中旧results不覆盖；新支持目录中PY、JSON、MD必须完整上传。
父目录的algorithms.py、channel_model.py、inner_solver.py、outer_solver.py、system_model.py均不需要替换。01至04也不需要替换。

## 提交
先让01完成起点筛选，日志出现FROZEN GEOMETRY READY后即可提交，无须等01全部蒙特卡洛结束。
默认冻结文件为 /home/lyq00008339/Lihuang_Li/toma/results/final_initial_geometry_i1200_o1500/frozen_initial_geometry.json 。
如已有该文件，直接提交；本程序不会重新筛选20起点。若文件位置不同，提交前设置HPC_SELECTION为它所在目录。

    cd /home/lyq00008339/Lihuang_Li/toma/05_communication_sensing_tradeoff
    sbatch submit_tradeoff.sh

网页提交可粘贴submit_tradeoff.sh完整内容。使用JM_CVX环境，申请1节点56核、48GB、48小时。48小时是作业上限，不是运行时间保证。

## 固定协议
20个权重点：0, 0.001, 0.005, 0.015, 0.03, 0.05, 0.075, 0.1, 0.125, 0.16, 0.18, 0.2, 0.24, 0.28, 0.32, 0.36, 0.42, 0.6, 0.8, 1。
每点场景8001至8100共100次蒙特卡洛，四方案共8000项独立任务；56进程动态领取，每进程BLAS单线程。尾部剩少量任务时无法占满56核。
内层最多1200，外层最多1500，绝对容差1e-4。每项只求解一次，不续跑、不复用旧结果、无跨权重点热启动。
Proposed和MA分别使用01筛选获胜的INITIAL几何。不是筛选得到的终态几何，也不读取终态资源；每项独立重建资源。
Proposed使用FP-aware，MA使用MA+FP-aware；两种固定阵列维持原布局。
16发16收：收发各4架子无人机，每架4阵元。主无人机20米；地面水平半径sqrt(1200)±10米、全方位均匀面积采样。
通信用户权重各为w/(K+J)，感知目标各为(1-w)/L。端点w=0或1不作额外目标或资源修饰。

## 输出
results/final_tradeoff_作业ID下保存逐场景CSV、均值CSV、收敛统计CSV、PNG/PDF/SVG、初末状态NPZ、冻结文件、日志和源码快照。
达到预算但未收敛的有效结果仍纳入均值；tradeoff_status.csv和completion.json单独列出未收敛数，图标题标注总收敛数。缺失任务不绘制部分场景均值。
结束后自动汇总all_data.json并生成同级final_tradeoff_作业ID_bundle.zip；一般非零退出也打包已完成数据，强制杀进程/节点故障无法保证执行退出钩子。
这张权重图单独打包，不参与原四图合并收集器。

冻结文件校验包括原筛选协议、模型及源文件身份，缺失或版本不匹配会停止，不退回默认起点。
--smoke仅为本地集成验证（1场景、3权重、2/2轮），结果明确标记smoke_only，不用于论文。正式提交脚本不使用此选项。
