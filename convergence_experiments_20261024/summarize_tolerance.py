"""Audit the o=1e-3/i=1e-5 rerun and report actual post-position FP work."""
import controlled_curvature as exp
from controlled_curvature import ROOT, source, np, save
from pathlib import Path
import json,csv,zipfile,hashlib
from system_model import build_geometry,ResourceState

OUT=ROOT/'tolerance_o1e3_i1e5'
def load(path):return json.loads(path.read_text(encoding='utf-8'))

def main():
    assert exp.production_hashes()==load(ROOT/'production_hashes.json')
    summary=[];table=[]
    with zipfile.ZipFile(source.ZIP) as z:
        for seed in [8003,8006]:
            stem=f's{seed}_p1_n120_initial_accelerated_joint'
            assert (OUT/f'{stem}_final.npz').exists(), 'Run has not finished'
            d=load(OUT/f'{stem}.json')
            # Preserve this older report's original 1e-6 reference even though
            # the full original files were subsequently replaced by request.
            ref=load(ROOT/'replaced_o1e4_i1e6_reference.json')
            old=next(r for r in ref['initial_runs'] if r['scene']==seed)
            old=dict(iterations=old['iterations'],seconds=old['seconds'],final_score=old['final_score'])
            assert d['inner_tolerance']==1e-5 and d['outer_tolerance']==1e-3
            assert d['converged'] and np.max(np.abs(np.diff(d['history'][-3:])))<=1e-3
            assert np.min(np.diff(d['history']))>=-1e-8
            cfg,_,_=source.load(z,'1',seed)
            with np.load(OUT/f'{stem}_final.npz') as state:
                g=build_geometry(state['endpoints'],cfg)
                r=ResourceState(**{n:state[n] for n in ['q_matrix','q_ul','b_ul','u_s']})
                assert state['previous_gradients'].shape==(8,3)
            source.e.check_state(g,r,cfg,{},'proposed')
            assert abs(source.e.metrics_values(g,r,cfg)['weighted_sum_rate']-d['final_score'])<1e-9
            calls=[];blocks=[]
            for i,row in enumerate(d['rows']):
                if i==0:stage='initial_resource_solve'
                else:
                    stage='after_position_update';calls.append(row['main_fp_steps_this_call'])
                    blocks.append(row['extrapolation_blocks'])
                table.append(dict(scene=seed,position_iteration=i if i else '',stage=stage,
                    fp_iterations=row['main_fp_steps_this_call'],extrapolation_blocks=row['extrapolation_blocks'],
                    inner_tolerance_met=row['main_inner_tolerance_met'],
                    gain=row['resource_gain'],seconds=d['inner_runs'][i]['seconds']))
            probe=d['post_final_position_probe']
            assert probe['included_in_solve_runtime'] is False
            calls.append(probe['fp_steps']);blocks.append(len(probe['rows']))
            probe_met=len(probe['rows'])>=2 and max(x['increment'] for x in probe['rows'][-2:])<=1e-5
            table.append(dict(scene=seed,position_iteration=d['iterations'],stage='final_position_diagnostic_only',
                fp_iterations=probe['fp_steps'],extrapolation_blocks=len(probe['rows']),
                inner_tolerance_met=probe_met,gain=probe['gain'],seconds=probe['seconds']))
            summary.append(dict(scene=seed,outer_iterations=d['iterations'],seconds=d['seconds'],
                final_wsr=d['final_score'],previous_outer_iterations=old['iterations'],
                previous_seconds=old['seconds'],previous_final_wsr=old['final_score'],
                wsr_change=d['final_score']-old['final_score'],
                wsr_change_percent=(d['final_score']/old['final_score']-1)*100,
                initial_fp_iterations=d['rows'][0]['main_fp_steps_this_call'],
                post_position_fp_iterations=calls,post_position_extrapolation_blocks=blocks,
                post_position_min=min(calls),post_position_max=max(calls),
                post_position_median=float(np.median(calls)),post_position_mean=float(np.mean(calls)),
                final_post_position_probe_fp=probe['fp_steps'],final_probe_gain=probe['gain'],
                main_fp_iterations=d['main_fp_steps'],all_candidate_fp_iterations=d['candidate_fp_steps'],
                accepted_endpoint_candidate_fp_counts=sorted({n for row in d['rows'] for n in row['endpoint_candidate_fp_iterations']}),
                final_outer_increments=np.diff(d['history'][-3:]).tolist()))
    save(OUT/'summary.json',dict(rows=summary,production_sources_unchanged=True))
    with (OUT/'inner_iterations.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    lines=['# 外层 1e-3、内层 1e-5：重跑结果', '',
        '沿用上一轮资源 FP 外推加速、原 PR+ 位置方向及 FP-aware-2 候选验收，在场景 8003、8006 的 30 dBm 原始冻结初态上独立重跑。仅修改测试阈值；正式源码、配置和冻结文件未改。', '',
        '外层以联合真实 WSR 绝对增量不超过 1e-3、连续两轮作为停止条件；内层加速器以接受的外推块目标增量不超过 1e-5、连续两块作为停止条件。每个外推块通常调用 3 次普通 FP，回溯时可能更多。内层每次最多实际执行 1200 次 FP。候选位置仍最多执行 2 次普通 FP。', '',
        '| 场景 | 外层轮数 | 耗时（秒） | 最终 WSR | 上次轮数 | 相对上次 WSR 变化 |',
        '|---|---:|---:|---:|---:|---:|']
    for s in summary:
        lines.append(f"| {s['scene']} | {s['outer_iterations']} | {s['seconds']:.2f} | {s['final_wsr']:.9f} | {s['previous_outer_iterations']} | {s['wsr_change_percent']:+.4f}% |")
    lines+=['', '## 更新位置后，内层实际需要多少次 FP', '',
        '下列顺序对应第 1 次位置更新后、第 2 次位置更新后……。每项均为下一次完整资源内层的实际 FP 次数，包含外推试探与被拒试探的工作；不把一个外推块当作一次 FP。初始资源求解单列，不混入“更新位置后”的统计。', '',
        '外层停止后额外执行一次只读诊断，从最终位置与资源的副本测量“若继续求解，下一次内层需要多少次 FP”。这次诊断不替换最终结果、不计入上述求解耗时，也不参与外层停止判定。', '']
    for s in summary:
        lines += [f"### 场景 {s['scene']}", '',
            f"- 初始资源求解：{s['initial_fp_iterations']} 次 FP。",
            '- 每次位置更新后的实际 FP 次数：`'+', '.join(map(str,s['post_position_fp_iterations']))+'`。',
            f"- 中位数 {s['post_position_median']:.1f} 次，均值 {s['post_position_mean']:.2f} 次，范围 {s['post_position_min']}–{s['post_position_max']} 次。",
            f"- 最后一个数是停止后的独立诊断：{s['final_post_position_probe_fp']} 次 FP，额外目标增益 {s['final_probe_gain']:.9g}。",
            f"- 两个最终外层增量：{s['final_outer_increments'][0]:.9g}、{s['final_outer_increments'][1]:.9g}，均不超过 1e-3。", '']
    lines += ['## 核验与解释', '',
        '真实 WSR 非下降、功率约束、接收波束单位范数、缆绳定长、防碰撞均已核验，最终 NPZ 重算 WSR 与日志一致。本次保存了最终 RCG 历史梯度与方向，便于后续定位。', '',
        '早期部分内层达到 1200 次上限，不等于达到内层精度；逐次是否满足阈值见 inner_iterations.csv。位置候选的少量 FP 与位置更新后的完整内层是两件事。', '',
        '本次外层阈值比上次 1e-4 更宽，内层 1e-5 也比上次实验的 1e-6 更宽（但比正式版本的 1e-4 更严）。因此新旧停止轮数和末值的差异包含精度变化，不能归因于新增算法加速；算法与上次相同。', '',
        '复现命令（分别使用 --seed 8003 和 --seed 8006）：', '', '```powershell',
        "& 'D:/miniconda/envs/toma_isac/python.exe' '../check_accelerated_joint.py' --initial --iterations 120 --seed 8003 --inner-tolerance 1e-5 --outer-tolerance 1e-3 --output-subdir tolerance_o1e3_i1e5 --final-probe", '```', '',
        'summary.json 为汇总；两个逐轮 JSON 保存每次内层全部外推块与实际 FP 数；inner_iterations.csv 按位置更新编号列出下一次内层工作量。']
    (OUT/'测试结论.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Refresh artifact hashes after the diagnostic driver gained instrumentation.
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in ROOT.rglob('*') if p.is_file() and p.suffix in ('.py','.json','.npz','.md','.png','.csv')
        and p.name!='artifact_manifest.json'}
    save(ROOT/'artifact_manifest.json',manifest)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
