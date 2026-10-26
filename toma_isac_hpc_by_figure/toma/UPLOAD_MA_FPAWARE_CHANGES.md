# 本次 MA 恢复 FP-aware：上传覆盖清单

完整包为 toma_MA_FPaware_no_extrap_o4_i5_all5_3nodes.zip；增量包为 toma_MA_FPaware_overwrite_patch.zip。旧的 plainfp 同名ZIP保留普通 MA 版本，请勿混用。
以下为相对于上一版ZIP新增或变化的文件，路径均相对服务器 toma/。

- `01_proposed_convergence/submit_convergence.sh`
- `02_wsr_vs_p_dl_max/submit_p_dl_max.sh`
- `03_wsr_vs_rho_si/submit_rho_si.sh`
- `04_wsr_vs_system_load/submit_system_load.sh`
- `05_communication_sensing_tradeoff/submit_tradeoff.sh`
- `05_communication_sensing_tradeoff/README_FINAL_TRADEOFF.md`
- `final_support_plainfp_o4_i5_s8005/figure_engine.py`
- `final_support_plainfp_o4_i5_s8005/ma_fp_aware.py`
- `final_support_plainfp_o4_i5_s8005/README.md`
- `final_support_plainfp_o4_i5_s8005/run_final.py`
- `tradeoff_support_plainfp_o4_i5_s8005/figure_engine.py`
- `tradeoff_support_plainfp_o4_i5_s8005/ma_fp_aware.py`
- `tradeoff_support_plainfp_o4_i5_s8005/README.md`
- `tradeoff_support_plainfp_o4_i5_s8005/reference_screening_identity.json`
- `tradeoff_support_plainfp_o4_i5_s8005/run_tradeoff_final.py`
- `multinode_support_plainfp_o4_i5/ma_aware_validation.json`
- `multinode_support_plainfp_o4_i5/README.md`
- `multinode_support_plainfp_o4_i5/run_multinode.py`
- `multinode_support_plainfp_o4_i5/validate_ma_aware.py`
- `multinode_support_plainfp_o4_i5/validation_report.json`
- `README.md`
- `README_CURRENT_PLAINFP_O4_I5.md`
- `submit_all_other_figures56.sh`
- `VALIDATION_PLAINFP_O4_I5.json`
- `UPLOAD_PLAINFP_O4_I5.json`
- `UPLOAD_MA_FPAWARE_CHANGES.md`（本清单）

必须上传两个支持目录中的 ma_fp_aware.py；新版实际导入调用，筛选身份包括这两个候选FP处理代码对应的有效模块。
先停止仍运行的旧版作业，再上传覆盖。Proposed和MA均使用FP-aware；候选位置最多两轮普通FP；全部无外推。
重新提交01进行本版筛选和收敛图计算。新筛选目录为 results/final_initial_geometry_plainfp_o4_i5_maaware_s8005；请先 unset HPC_SELECTION HPC_OUTPUT，勿复用旧筛选结果。
