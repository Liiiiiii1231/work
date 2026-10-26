# 上传到 toma/test 的10次MC试跑包

## 放置方式

ZIP内只有test目录。把test中的文件放入平台现有目录：

    /home/lyq00008339/Lihuang_Li/toma/test

也可以在toma目录下解压ZIP，得到同级的test目录。不要放成toma/test/test。
若test内有之前上传的独立版本algorithms.py、channel_model.py、inner_solver.py、outer_solver.py、system_model.py，请先将它们移出test，避免遮蔽父目录模块。
本包不包含这5个共享文件。

## 哪些文件需要替换

父目录toma中的任何文件都不需要替换。
本地逐文件核查确认：algorithms.py、channel_model.py、inner_solver.py、outer_solver.py、system_model.py与原HPC版本一致。
服务器内容未直接读取；运行实际采用父目录文件，实际源码哈希会记录在结果metadata中。

test/figure_engine.py：试验专用入口逻辑，包含地面圆环场景采样；不覆盖toma/figure_engine.py。
test/run_figures.py：试跑调度、结果校验、绘图，优先使用test/figure_engine.py，再从父目录导入5个模型/求解器。
test/tradeoff_config.json：16发16收、10场景和原20权重。父目录配置不变。
test/finalize_run.py：结束后汇总并打包，还会保存实际使用的父目录5个共享文件快照。
test/submit_tradeoff_mc10.sh：平台提交脚本，默认工作目录已改成toma/test。
其余文件是配置生成工具、短程验证、模拟设置说明和校验清单。

## 已确认的设置

主无人机高20米；地面相对坐标z=-20米。
地面水平半径24.64101615137755至44.64101615137755米，围绕正下方覆盖0至360度。
5DL、5UL、5目标独立按圆环面积均匀采样，同场景跨权重和方案共用位置。
收发各4架子无人机，每根1米缆绳上4个等距阵元，即16发16收。
感知因子列数保持原配置（5目标基准为4）；功率、噪声、RCS、求解器和停止容差不变。
权重图场景8001至8010；20权重×10场景×4方案=800个独立任务，无跨权重热启动。

## 提交

    cd /home/lyq00008339/Lihuang_Li/toma/test
    sbatch submit_tradeoff_mc10.sh

默认10核、10个动态取任务的单线程进程，24GB、48小时，cpu分区，JM_CVX环境。
若使用50核：

    sbatch --cpus-per-task=50 --mem=48G submit_tradeoff_mc10.sh

结果位置：

    toma/test/results/tradeoff_mc10_作业ID/
    toma/test/results/tradeoff_mc10_作业ID/all_data.json
    toma/test/results/tradeoff_mc10_作业ID_bundle.zip

all_data.json包括表格、JSON记录、NPZ数值、日志和源码配置快照。
ZIP还包含原始文件和图表。Python正常或报错返回后自动导出；平台强制终止可能需要手动运行：

    python finalize_run.py --output /绝对路径/test/results/tradeoff_mc10_作业ID --run-exit-code 1

续跑保持本地与父目录源码不变：

    HPC_OUTPUT=/绝对路径/test/results/tradeoff_mc10_原作业ID sbatch submit_tradeoff_mc10.sh

身份校验覆盖test内Python文件及父目录5个共享文件。不要复用旧独立工程的结果目录。
已通过所有方案/权重代表性初态检查、10场景位置校验、12项两进程短程优化和汇总打包检查。
短程验证不代表已完成10次MC。未代用户提交平台作业。
