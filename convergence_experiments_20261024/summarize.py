"""Audit experiment outputs and write the human-readable Chinese report."""
import controlled_curvature as exp
from controlled_curvature import ROOT, np, source, save
import json,hashlib,zipfile,io
from dataclasses import replace

def load(name): return json.loads((ROOT/name).read_text(encoding='utf-8'))

def main():
    assert exp.production_hashes()==load('production_hashes.json')
    controlled=[]
    for power in ['0.1','1','10']:
        base=load(f's8003_p{power}_n30_budget_baseline.json')
        frozen=load(f's8003_p{power}_n30_budget_resource_only.json')
        assert base['fp_iterations']['main']+sum(r['accepted_candidate_fp_iterations'] for r in base['rows'])==510
        assert frozen['fp_iterations']['main']+frozen['fp_iterations']['candidate']==510
        controlled.append(dict(power_dBm=30+10*np.log10(float(power)),
            baseline_gain=base['gain'],fixed_geometry_gain=frozen['gain'],
            retained_gain_percent=frozen['gain']/base['gain']*100,
            baseline_seconds=base['seconds'],fixed_geometry_seconds=frozen['seconds']))
    accelerated=[]
    for file in sorted(ROOT.glob('resource_acceleration_p*_b1500*.json')):
        d=load(file.name); a=d['accelerated'];b=d['plain']
        assert a['fp_steps']==b['fp_steps']==1500
        assert min(r['increment'] for r in a['rows'])>=-1e-8
        reached=next((r for r in a['rows'] if r['wsr']>=b['final_score']),None)
        accelerated.append(dict(scene=d['scene'],power_dBm=30+10*np.log10(d['power_W']),
            plain_gain=b['gain'],accelerated_gain=a['gain'],gain_ratio=a['gain']/b['gain'],
            plain_seconds=b['seconds'],accelerated_seconds=a['seconds'],
            fp_steps_to_plain_final=reached['fp_steps'] if reached else None,
            seconds_to_plain_final=reached['seconds'] if reached else None))
    local=load('s8003_p1_n30_accelerated_joint.json')
    tight=load('s8003_p1_n30_budget_baseline_tight.json')
    initial=load('s8003_p1_n30_initial_baseline.json')
    initial_tight=load('s8003_p1_n5_budget_initial_baseline_tight.json')
    initial_acc=load('s8003_p1_n30_initial_accelerated_joint.json')
    # Longer controlled runs must have finished and saved final states.
    long=[]
    for seed in [8003,8006]:
        stem=f's{seed}_p1_n120_initial_accelerated_joint'
        assert (ROOT/f'{stem}_final.npz').exists(), f'{seed} experiment still running'
        d=load(stem+'.json');assert np.min(np.diff(d['history']))>=-1e-8
        assert d['inner_tolerance']==1e-5 and d['outer_tolerance']==1e-4
        if d['converged']: assert np.max(np.abs(np.diff(d['history'][-3:])))<=1e-4
        with zipfile.ZipFile(source.ZIP) as z:
            cfg,g,r=source.load(z,'1',seed)
            with np.load(ROOT/f'{stem}_final.npz') as state:
                from system_model import build_geometry,ResourceState
                gg=build_geometry(state['endpoints'],cfg)
                rr=ResourceState(**{n:state[n] for n in ('q_matrix','q_ul','b_ul','u_s')})
            source.e.check_state(gg,rr,cfg,{},'proposed')
            recomputed=source.e.metrics_values(gg,rr,cfg)['weighted_sum_rate']
            assert abs(recomputed-d['final_score'])<1e-9
        long.append(dict(scene=seed,iterations=d['iterations'],converged=d['converged'],
            inner_tolerance=d['inner_tolerance'],outer_tolerance=d['outer_tolerance'],
            final_score=d['final_score'],seconds=d['seconds'],
            main_fp_steps=d['main_fp_steps'],candidate_fp_steps=d['candidate_fp_steps'],
            last_increments=np.diff(d['history'][-3:]).tolist()))
    # Outer tolerance alone must not change the common prefix of the reruns.
    for seed in [8003,8006]:
        repeated=load(f's{seed}_p1_n120_initial_accelerated_joint.json')
        loose=load(f'tolerance_o1e3_i1e5/s{seed}_p1_n120_initial_accelerated_joint.json')
        n=min(len(repeated['history']),len(loose['history']))
        assert np.array_equal(repeated['history'][:n],loose['history'][:n])
    # Historical server trajectories are references, not local equal-time controls.
    with zipfile.ZipFile(source.ZIP) as z:
        with np.load(io.BytesIO(z.read('final_33774/candidate_0/tasks/convergence/1/scene_8003/proposed/call_001.npz'))) as a:
            historical_difference=float(np.max(abs(a['wsr_history'][:31]-initial['history'])))
    total_tight=sum(tight['fp_iterations'].values());total_acc=local['main_fp_steps']+local['candidate_fp_steps']
    summary=dict(controlled=controlled,resource_acceleration=accelerated,initial_runs=long,
        local_main_fp_reduction_percent=(1-total_acc/total_tight)*100,
        local_tight_plain_gain=tight['gain'],local_accelerated_gain=local['gain'],
        local_initial_30_gain=initial['gain'],initial_baseline_replay_max_difference_from_server=historical_difference,
        source_unchanged=True,initial_acceleration_same_inner_prefix_exact_repeat=True,
        replaced_reference=load('replaced_o1e4_i1e6_reference.json'))
    save(ROOT/'summary.json',summary)
    lines=[
        '# Proposed 收敛长尾：第二阶段受控实验',
        '',
        '结论：后期增益的重要来源是资源 FP 的慢迭代，而不只是位置方向与跨端点曲率。资源外推加速是目前更有希望的独立实验方案；正式源码、配置和冻结文件均未修改。小规模结果不构成全部场景的完整收敛验证。',
        '',
        '## 实验边界与复现',
        '',
        '- 使用原 final_33774_bundle.zip；配置摘要、保存末态及初态的 SHA256 按旧加载器核验。所有对照独立初始化，不跨权重或参数传递资源。',
        '- 末态实验一致重置 RCG 历史，不能称作原 1500 轮轨迹的精确续跑。从 initial.npz 开始的实验使用原冻结初始几何与资源。',
        '- 原算法基准保持内外绝对 1e-4；早期 30 轮资源加速与 tight 对照使用内层 1e-6。最新两项冻结初态测试已按用户要求覆盖为内层 1e-5、外层绝对 1e-4 连续两轮。候选仍最多两轮 FP；未使用窗口平台早停。',
        '- 120 轮是两项独立实验的预算上限，低于正式 1500 轮；不续算旧作业，也不以更换场景或放宽阈值制造收敛。',
        '- BLAS 单线程。部分独立实验同时运行，因此本机时间是实测参考值；实际 FP 次数同时报告，不能将秒数比例视为精确性能定律。',
        '- 补充场景按收敛结果包已有编号顺序选取 8006、8016，未按速度筛选。8002 不在该历史 30 场景列表，未对其运行或用其替换慢样本。',
        '',
        '运行目录为本文件所在目录，Python 为 D:/miniconda/envs/toma_isac/python.exe。脚本仅在当前进程中替换停止量计算或实验分支，不写正式模块。脚本依赖旧诊断目录 check_larger_step.py，迁移时需同时保留该加载器与工程目录。',
        '',
        '```powershell',
        "& 'D:/miniconda/envs/toma_isac/python.exe' ./controlled_curvature.py --powers 0.1 1 10 --methods baseline resource_only --iterations 30 --fixed-budget",
        "& 'D:/miniconda/envs/toma_isac/python.exe' ./resource_acceleration.py --budget 1500",
        "& 'D:/miniconda/envs/toma_isac/python.exe' ./check_accelerated_joint.py --initial --iterations 120 --seed 8003",
        "& 'D:/miniconda/envs/toma_isac/python.exe' ./check_accelerated_joint.py --initial --iterations 120 --seed 8006",
        '```',
        '',
        '## 1. 资源更新与位置更新的拆分',
        '',
        '场景 8003 的三个保存末态各运行 30 轮。原算法每轮主内层均只执行 1 次 FP，8 个已接受端点各执行 2 次 FP，即沿接受路径实际推进资源 17 次；固定几何对照也每轮执行 17 次，总计均为 510 次。原算法被拒候选仍需额外计算，均已计入原基准总预算。固定几何对照只用于定位瓶颈，不作为替代 Proposed 的正式方案。',
        '',
        '| 功率 | 原算法增益 | 固定几何增益 | 增益比 | 原算法/固定几何耗时（秒） |',
        '|---|---:|---:|---:|---:|']
    for row in controlled:
        lines.append(f"| {row['power_dBm']:.0f} dBm | {row['baseline_gain']:.9f} | {row['fixed_geometry_gain']:.9f} | {row['retained_gain_percent']:.2f}% | {row['baseline_seconds']:.2f} / {row['fixed_geometry_seconds']:.2f} |")
    lines += ['',
        '这些是两条不同轨迹的增益比，不能解释成原轨迹中资源与几何贡献的因果百分比。逐候选的固定资源损失与 FP 恢复往往幅度较大且相互抵消，也不能只把整个位置阶段净增益都计作几何优化收益。',
        '',
        '## 2. 曲率预处理未解决问题',
        '',
        '以严格执行两轮 FP 的候选映射计算切空间数值梯度与 Hessian；端点块预处理与完整耦合预处理均使用特征值截断正定化、球面回缩、碰撞检查及真实 WSR Armijo 验收。曲率每 10 轮更新，探针 FP 全部计入成本。',
        '',
        '在原内层精度下，两种方案都在第 3 轮满足原外层停止条件，但仅增益 0.000246 / 0.000396，明显低于原算法 30 轮的 0.006634。因此拒绝将这称为长尾已解决。资源阶段收紧至 1e-6 后，完整曲率方案 30 轮增益约 0.117432、耗时约 186 秒，与 tight 普通 FP 的 0.117492、182 秒接近，未表现出独立加速价值。',
        '',
        '## 3. 资源 FP 外推：等实际计算预算',
        '',
        '在固定保存几何上，取两次 FP 的发射变量映射 x1=T(x0)、x2=T(x1)，r=x1−x0、v=x2−2x1+x0，试探 x0+2ar+a²v，a 在 [1,64] 内按 ||r||/||v|| 选择。向下行总功率球和上行幅度区间投影，再执行一次 FP 重新适配资源。只有真实 WSR 不低于普通两步 FP 结果时接受，否则缩小外推量或退回普通结果。被拒试探也计入实际 FP 次数。',
        '',
        '每个实验的普通 FP 与加速 FP 都执行 1500 次实际 FP 更新；不以调用次数替代迭代数。此处比较目标增益，不宣称达到驻点。',
        '',
        '| 场景/功率 | 普通 FP 增益 | 加速 FP 增益 | 增益倍率 | 普通/加速耗时（秒） |',
        '|---|---:|---:|---:|---:|']
    for row in accelerated:
        lines.append(f"| {row['scene']} / {row['power_dBm']:.0f} dBm | {row['plain_gain']:.6f} | {row['accelerated_gain']:.6f} | {row['gain_ratio']:.2f} | {row['plain_seconds']:.2f} / {row['accelerated_seconds']:.2f} |")
    lines += ['',
        '## 4. 接回原位置更新的对照',
        '',
        f"场景 8003、30 dBm 保存末态独立重启。主资源内层采用外推加速、精度 1e-6，位置仍用原 PR+ 和 FP-aware-2。30 轮增益 {local['gain']:.9f}，耗时 {local['seconds']:.2f} 秒，实际 FP {total_acc} 次。tight 普通 FP 对照同为 30 轮，增益 {tight['gain']:.9f}，耗时 {tight['seconds']:.2f} 秒，实际 FP {total_tight} 次；FP 次数减少 {(1-total_acc/total_tight)*100:.2f}%。两者均未满足外层停止条件。",
        '',
        f"原始冻结初态的前 5 轮，同样收紧内层精度时，普通 FP 的 WSR={initial_tight['final_score']:.9f}，主 FP 6000 次；加速 FP 的 WSR={initial_acc['history'][5]:.9f}，主 FP 同为 6000 次。计入候选后总 FP 分别为 {sum(initial_tight['fp_iterations'].values())} 和 {initial_acc['rows'][4]['main_fp_steps']+initial_acc['rows'][4]['candidate_fp_steps']} 次。该对照支持加速收益并非只由收紧内层阈值带来。",
        '',
        '## 5. 最新覆盖结果：原始冻结初态，外层 1e-4、内层 1e-5',
        '',
        '原先内层 1e-6 的两项完整测试 JSON、最终 NPZ 和本报告对应曲线均已被本次重跑覆盖；仅在 replaced_o1e4_i1e6_reference.json 保留原标量结果供对照。其余早期局部诊断未覆盖。',
        '',
        '| 场景（30 dBm） | 实际轮数 | 满足原外层阈值 | 最终 WSR | 耗时（秒） | 实际 FP 次数 |',
        '|---|---:|---|---:|---:|---:|']
    for row in long:
        lines.append(f"| {row['scene']} | {row['iterations']} | {'是' if row['converged'] else '否'} | {row['final_score']:.9f} | {row['seconds']:.2f} | {row['main_fp_steps']+row['candidate_fp_steps']} |")
    lines += ['',
        '所有已保存轨迹的真实 WSR 均通过非下降检查，每次接受后检查功率、单位范数、缆绳定长与防碰撞；两项较长实验还从最终 NPZ 重算真实 WSR 并核对。该检查不构成约化目标梯度趋零的证明。',
        '',
        f"跨平台复现限制：本地原算法从相同 initial.npz 开始的前 30 轮，与服务器保存轨迹最大 WSR 差约 {historical_difference:.6f}，虽然五个正式模块与服务器快照归一化换行后相同。初值一致但后续轨迹有差异，尚未进一步定位；因此服务器旧轨迹仅作历史参考，收益与耗时结论以本机成对对照为准。本次与内层同为 1e-5、外层 1e-3 的测试在共同迭代区间内轨迹逐值完全相同。",
        '',
        '## 正式更新决策',
        '',
        '暂不提升为正式版本。资源 FP 外推已显示值得继续验证的效果，当前证据不足以宣布所有 Proposed 长尾问题解决；尤其需要更广泛的初态、功率、权重及完全停止轨迹检查，确认单调验收之外的最终优化质量。若推广，应使用新版本目录，记录完整算法参数与 RCG 历史、同步源码身份及冻结几何校验，并按新求解器重新审视起点筛选；不能覆盖现行作业依赖。',
        '',
        '未使用邻近权重点热启动，未增加正式迭代上限，未放宽外层阈值，未重跑整张图。',
        '',
        '原始 JSON、最终 NPZ、脚本和 production_hashes.json 均在本目录；summary.json 为机器可读汇总。']
    (ROOT/'实验结论.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    for power,label in [('0.1','20 dBm'),('1','30 dBm'),('10','40 dBm')]:
        d=load(f'resource_acceleration_p{power}_b1500.json'); b=d['plain'];a=d['accelerated']
        line,=axes[0].plot(np.arange(len(b['history'])),np.array(b['history'])-b['initial_score'],ls='--',alpha=.75)
        axes[0].plot([r['fp_steps'] for r in a['rows']],[r['wsr']-a['initial_score'] for r in a['rows']],color=line.get_color(),label=label)
    axes[0].set(xlabel='Actual FP iterations (including rejected trials)',ylabel='WSR gain',title='Fixed geometry: solid = extrapolated; dashed = plain')
    axes[0].legend();axes[0].grid(alpha=.25)
    for seed in [8003,8006]:
        d=load(f's{seed}_p1_n120_initial_accelerated_joint.json')
        axes[1].plot(range(1,len(d['history'])),d['history'][1:],label=f'Accelerated, scene {seed}')
    axes[1].plot(range(1,len(initial['history'])),initial['history'][1:],ls='--',label='Original local control, scene 8003')
    axes[1].set(xlabel='Outer iteration (initial WSR excluded)',ylabel='WSR',
        title='Original frozen initial states; outer abs tol 1e-4, twice\nInner tol: accelerated 1e-5; original 1e-4')
    axes[1].legend(fontsize=8);axes[1].grid(alpha=.25)
    fig.tight_layout();fig.savefig(ROOT/'diagnostic_comparison.png',dpi=180)
    manifest={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in ROOT.iterdir() if f.is_file() and f.suffix in ('.py','.json','.npz','.md','.png') and f.name!='artifact_manifest.json'}
    save(ROOT/'artifact_manifest.json',manifest)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
