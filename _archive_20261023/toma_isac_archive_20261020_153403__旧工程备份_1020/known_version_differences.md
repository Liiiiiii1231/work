# 已知版本差异（仅记录，未修复）

- 当前 run_all_figures_unified.py 的 __main__ 实际调用 main25。
- main25 默认 10 个蒙特卡洛场景。
- 收敛图固定使用场景 8001。
- tradeoff 复用旧冻结结果，不是重新执行 100 场景权重图。
- README_HPC.md 描述的是另一个 100 场景方案。
- hpc/run_array.sh 传递 --scene-index，但当前 main25 入口不支持该参数。
- 本次未替换入口、修复协议、修改配置或哈希、放宽 checkpoint 校验，也未提交 HPC 作业。
