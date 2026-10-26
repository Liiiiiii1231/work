# 本次覆盖清单：MA恢复原始 MA + FP

压缩包沿用用户指定文件名；内容已修改为Proposed FP-aware、MA普通FP、无外推。
将压缩包内 toma/ 全部内容合并覆盖到服务器 toma/。下列为相对修改前压缩包新增或变化的文件：

- `01_proposed_convergence/submit_convergence.sh`
- `02_wsr_vs_p_dl_max/submit_p_dl_max.sh`
- `03_wsr_vs_rho_si/submit_rho_si.sh`
- `04_wsr_vs_system_load/submit_system_load.sh`
- `05_communication_sensing_tradeoff/README_FINAL_TRADEOFF.md`
- `05_communication_sensing_tradeoff/submit_tradeoff.sh`
- `FINAL_CONFIGURATION.md`
- `final_support_plainfp_o4_i5_s8005/figure_engine.py`
- `final_support_plainfp_o4_i5_s8005/README.md`
- `final_support_plainfp_o4_i5_s8005/run_final.py`
- `multinode_support_plainfp_o4_i5/ma_original_validation.json`
- `multinode_support_plainfp_o4_i5/README.md`
- `multinode_support_plainfp_o4_i5/run_multinode.py`
- `multinode_support_plainfp_o4_i5/validate_ma_original.py`
- `multinode_support_plainfp_o4_i5/validation_report.json`
- `README.md`
- `README_CURRENT_PLAINFP_O4_I5.md`
- `submit_all_other_figures56.sh`
- `tradeoff_support_plainfp_o4_i5_s8005/figure_engine.py`
- `tradeoff_support_plainfp_o4_i5_s8005/README.md`
- `tradeoff_support_plainfp_o4_i5_s8005/reference_screening_identity.json`
- `tradeoff_support_plainfp_o4_i5_s8005/run_tradeoff_final.py`
- `UPLOAD_PLAINFP_O4_I5.json`
- `VALIDATION_PLAINFP_O4_I5.json`
- `UPLOAD_FINAL_MAFP_CHANGES.md`（本清单）

残留的旧 ma_fp_aware.py 不会被本版 MA 调用，不必手动删除。
重新提交01，在新的 mafp_final 筛选目录生成本版筛选结果；勿复用 MA FP-aware 结果。
