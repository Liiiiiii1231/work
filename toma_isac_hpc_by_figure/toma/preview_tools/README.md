# 当前作业的只读阶段图

上传preview_tools到平台toma/preview_tools，不放入旧toma/test，避免改变旧工程的源码身份。
不修改旧源码、配置、检查点或收敛标记，不调用优化器。可与当前作业同时运行。

网页新提交作业时粘贴submit_preview.sh内容即可；或者：
    cd /home/lyq00008339/Lihuang_Li/toma/preview_tools
    sbatch submit_preview.sh

默认读取toma/test/results/tradeoff_mc10_33614，输出到其同级tradeoff_mc10_33614_preview_新作业ID目录及ZIP。
SOURCE不同可用HPC_SOURCE环境变量指定。
只读取record指定的已保存状态，并核验initial/state哈希及目标历史一致性。
只有同一权重下四方案均有全部10个预期场景，才进入平均曲线；已保存未收敛结果明确计入并用空心标记。
缺失任务不使用初态/零值补齐，不筛除慢场景；不完整权重列在preview_summary.json。
saved_per_scene.csv含所有读取到的状态，preview_mean.csv含可画的平均值及收敛数量。
当前进程尚未保存的迭代无法从磁盘恢复到图中；工具不会要求当前程序停止或跳过正在执行的调用。
上传样本验证：269份记录，263份收敛，前6个权重数据完整，绘图通过。
