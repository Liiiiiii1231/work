# results 清理评估（仅检查，未移动或删除）

检查范围：`D:\Vs codeProjects\research work1\toma_isac\results`。当前总大小约 69.65 MiB。本次没有执行仿真、修改源码/配置、移动或删除 results 文件。

## 建议优先移出项目归档的历史实验目录

| 相对 results 的路径 | 文件数 | 大小 MiB |
| --- | ---: | ---: |
| `auto_figure_runs/` | 54 | 2.212 |
| `old_init_power_pilot/` | 133 | 0.637 |
| `regular025_multistart_pilot/` | 5 | 0.010 |
| `stage10/cold20_ms20_in1e6_out1e5_continued/` | 163 | 7.808 |
| `stage10/multistart20_probe/` | 44 | 0.173 |
| `stage10/proposed_0125_trace/` | 244 | 1.905 |
| `stage10/regular_ul_init_compare_representative/` | 31 | 0.070 |
| `stage10/four_schemes_top10_abs1e4_scene8001/` | 455 | 1.812 |
| `stage10/top10_abs1e4_scene8001/` | 468 | 2.130 |
| `stage10/top10_abs1e4_scene8001_b6_09_continued/` | 11 | 0.359 |

这些目录对应旧绘图批次、初始化对比、诊断、冷启动及种子探索等实验；相关历史驱动已归档，当前保留入口未发现把这些目录作为必要输入读取。仍建议移动归档保留历史，不永久删除。

## 经逐文件大小及 SHA256 比对，ZIP 内每个文件均有相同本地副本

| ZIP 路径 | 大小 MiB |
| --- | ---: |
| `paper_figures_regular_abs_v2.zip` | 8.529 |
| `regular25_scene8001_20to40.zip` | 0.251 |
| `paper_figures_regular_abs_v2/source_configs.zip` | 0.007 |
| `paper_figures_regular_abs_v2/tables.zip` | 0.050 |
| `stage10/fixed_ul025mW_MA_transition4.zip` | 0.241 |
| `stage10/proposed_0125_trace.zip` | 0.993 |
| `stage10/regular_planar_pm_fixed_ul025mW_refined16.zip` | 0.356 |
| `stage10/regular_planar_pm_full27.zip` | 0.503 |
| `stage10/regular_ul_init_compare_representative.zip` | 0.048 |
| `stage10/seed2042_refined_weights.zip` | 0.790 |

同名不能直接当作重复：上述 10 项已核对压缩包每个文件在本地具有大小及 SHA256 相同的副本。可以优先移动这些 ZIP 归档；不要因此删除所对应的正式解压目录。

## repair_backup 可作为第二批归档候选

- `paper_figures_regular25_multistart/repair_backup/`：345 文件，1.332 MiB。
- `paper_regular25_convergence/repair_backup/`：38 文件，0.146 MiB。
- `paper_regular25_rho_si/repair_backup/`：377 文件，1.360 MiB。

目前四类 25 mW 正式结果 completion 均为完成；活动 record 的 initial.npz 和当前 state 哈希全部有效。repair_backup 为以前初始化修复的追溯副本，当前正常续跑不读取；可以整体归档到项目外，但不建议直接删掉唯一修复记录。

## 必须继续保留或暂缓清理

- `paper_figures_regular25_multistart/`
- `paper_regular25_convergence/`
- `paper_regular25_rho_si/`
- `paper_regular25_system_load/`
- `regular25_multistart_pilot/`
- `regular25_scene8001_20to40/`
- `paper_figures_regular_abs_v2/`
- `stage8/`
- `stage10/fixed_ul025mW_MA_transition4/`
- `stage10/seed2042_refined_weights/`
- `stage10/regular_planar_pm_fixed_ul025mW_refined16/`
- `stage10/regular_planar_pm_full27/`
- `stage10/regular_planar_pm_refined20/`
- `stage10/fixed_ul025mW_MA_add4.zip`
- `stage10/stage10_tradeoff.csv`
- `stage10/stage10_tradeoff.npz`
- `stage10/stage10_failures.csv`

其中四个 paper_*25 正式目录分别保留全部 candidate、record、metadata、initial、状态、CSV 和图片；两个 regular25 目录是统一入口默认复用来源。

旧文件夹也不能一概移走：当前论文图仍与 stage8 和 paper_figures_regular_abs_v2 内图片字节匹配。本文不改变论文引用，也未认定旧权重所有版本都已被新图完全替代。

## ZIP 特别例外

- `stage10/fixed_ul025mW_MA_add4.zip`：没有对应完整解压目录，是已接受 MA add4 结果的打包副本，保留。
- `stage10/regular_ul_init_compare018.zip`：没有本地等价解压副本；可随历史对比实验整体归档，但不是冗余包。
- `stage10/four_schemes_top10_abs1e4_scene8001.zip`：包内 458 文件，只有 455 文件有相同本地副本，另 3 张诊断图仅在包内；可作历史实验归档，不能按完全重复直接丢弃。

## 暂不清理单个状态文件

活动正式目录和两个复用目录共发现 36 个较早的 call_*.npz 没被最新 record 直接引用，总计仅约 0.445 MiB。当前断点完整，节省空间很少，建议保持原样，避免破坏追溯。candidate 不是缓存，不按名字、扩展名或“已收敛”状态清理。

## 建议批次与预计大小

- 历史实验目录：17.11 MiB。
- 已核对的重复 ZIP：11.77 MiB。
- 修复备份（第二批）：2.84 MiB。
- 以上仅为待审阅建议。正式执行仍需先检查运行进程，使用项目同级归档、保留相对路径并建立逐文件 SHA256 清单；不在本次分析中执行移动。
