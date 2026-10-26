# 收敛长尾：自己排查时看哪些文件

当前正式版本为final_support_i1200_o1500_s8005及tradeoff_support_i1200_o1500_s8005。诊断脚本与结果在 _archive_20261023/convergence_33774_audit_20261024__收敛长尾诊断。正式求解器和运行配置尚未修改。

## 建议阅读顺序

1. outer_solver.py：首先看armijo_endpoint_update_fp_aware（502行）。初始alpha取cfg.initial_outer_step，拒绝后乘backtracking_factor；alpha不是以米为单位的实际位移。记录接受alpha、回溯次数、candidate_fp_calls、位移和目标变化。
2. 同一文件central_difference_gradient（143行）：finite_diff_delta控制中央差分；换1e-6、1e-7、1e-8比较投影后的梯度范数和方向，不能只比较原始三维欧氏梯度。
3. 同一文件compute_rcg_direction（248行）：看PR+的beta、方向与梯度的夹角，以及是否重启。beta大不一定错误，但方向过长或近乎垂直时应做重启对照。
4. 同一文件update_positions_once_fp_aware（646行）：依次更新端点；这里已有diagnostic_records参数，可收集每端点FPAwareEndpointDiagnostic。不需要修改正式函数才能做局部诊断。
5. algorithms.py：run_joint_algorithm（54行）先做完整资源更新，再做位置更新；342行之后计算停止量，要求连续两次小于阈值。FP-aware候选最多2轮由调用位置传入。
6. final_support_i1200_o1500_s8005/figure_engine.py：54行定义abs_change，158行run_solver_call临时把模块内relative_change替换为绝对增量。因此仅看system_model.py里relative_change的原定义会误判实际判据。311行pair负责几何和资源初始化。
7. final_support_i1200_o1500_s8005/final_config.json：顶部protocol.protocol设置1200/1500及绝对容差1e-4；模型stage8_monte_carlo.algorithm中有initial_outer_step=0.1、finite_diff_delta=1e-6、backtracking_factor=0.5、max_backtracking_iter=30。确认使用的是base_config指向的模型，不要只改第一个同名字段。
8. inner_solver.py：solve_inner_problem（890行）和1055行附近停止条件。查看每次实际内层轮数、内层前后目标差。候选FP只做少量轮数，与每轮外层开头完整内层求解不同。

权重图用tradeoff_support_i1200_o1500_s8005/figure_engine.py及tradeoff_final_config.json，不能只改四图支持目录。修改求解器或配置会影响冻结文件身份校验；不要在正在运行的作业上直接覆盖源码。

## 下载结果后先看

- completion.json：complete表示任务完成；all_converged才表示全部满足停止条件，两者不同。
- tables/convergence_per_scene.csv：逐场景iterations、converged、runtime_s，先查是否统一碰到上限。
- tables/convergence_histories.csv：逐场景、逐功率、逐轮WSR，计算相邻增量和后100/500轮累计增益。不要只看均值PNG。
- candidate_0/tasks/convergence/功率/scene_编号/proposed/record.json：effective_config是真实参数，initial_sha256和state_sha256用于校验。
- 同目录initial.npz与call_001.npz：初末几何、资源、WSR历史，可复现局部诊断。当前不保存RCG历史，所以从末态重启不等于精确续算原轨迹。
- frozen_initial_geometry.json：获胜起点编号、筛选末值、是否收敛，以及冻结INITIAL几何；不是终态位置。
- provenance/：服务器本次真实源码快照。排查已完成作业时优先看它，不能假定与本地最新文件始终一致。

## 本次可直接复用的检查

- check_larger_step.py、larger_step_results.csv/json：同一保存状态，多档初始步长的单轮对比。
- check_step_trajectories.py、larger_step_30_iteration_comparison.json：0.1、0.8、1.0三个初始alpha的30轮轨迹对比。
- check_gradient_direction_curvature.py、gradient_curvature_results.json：5个状态共40个端点的差分尺度、切向梯度和局部曲率。
- rcg_direction_comparison.json：相同状态出发的30轮RCG与每轮重启梯度方向对照。
- candidate_step_fp_budget_scan.json：单端点增大实际位移及2/10/50候选FP上限对照。

Windows PowerShell复现（需要原始final_33774_bundle.zip仍在Downloads）：

    & 'D:/miniconda/envs/toma_isac/python.exe' 'D:/Vs codeProjects/research work1/_archive_20261023/convergence_33774_audit_20261024__收敛长尾诊断/check_gradient_direction_curvature.py'

脚本只读正式模块，结果写入诊断目录；会覆盖该脚本自己生成的诊断JSON。它们用于小规模检查，不运行整张图。

## 解释指标时注意

- 梯度误差使用更小差分尺度的估计作为参照，不是解析真值。
- 曲率探针固定候选FP为准确2轮，包括零位移点，以免提前停止导致假曲率。正式算法仍使用原来的最多2轮。
- 如果比较初始步长，必须固定同一状态、资源、方向历史初始化和停止精度；比较实际接受位移、单位时间增益，不只比较初始alpha。
- 不能把本地数十轮试验当作全程收敛轮数或全局最优证明。

补充检查：check_joint_curvature.py / joint_tangent_curvature.json 用于跨端点耦合；direction_curvature_summary.json及方向与曲率检查结论.md汇总本次发现。
