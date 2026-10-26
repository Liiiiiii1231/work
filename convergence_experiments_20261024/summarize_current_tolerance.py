"""Current root test: outer 1e-4, inner 1e-5; replaces the 1e-6 runs."""
import controlled_curvature as exp
from controlled_curvature import ROOT,source,np,save
import csv,json,hashlib,zipfile
from system_model import build_geometry,ResourceState

def load(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    assert exp.production_hashes()==load(ROOT/'production_hashes.json')
    oldref=load(ROOT/'replaced_o1e4_i1e6_reference.json')
    out=[];calls=[]
    with zipfile.ZipFile(source.ZIP) as z:
        for seed in [8003,8006]:
            stem=f's{seed}_p1_n120_initial_accelerated_joint'
            d=load(ROOT/f'{stem}.json')
            loose=load(ROOT/'tolerance_o1e3_i1e5'/f'{stem}.json')
            old=next(r for r in oldref['initial_runs'] if r['scene']==seed)
            assert d['inner_tolerance']==1e-5 and d['outer_tolerance']==1e-4
            assert np.min(np.diff(d['history']))>=-1e-8
            if d['converged']:assert max(abs(np.diff(d['history'][-3:])))<=1e-4
            assert d['iterations']>loose['iterations']
            assert np.array_equal(d['history'][:len(loose['history'])],loose['history'])
            cfg,_,_=source.load(z,'1',seed)
            with np.load(ROOT/f'{stem}_final.npz') as a:
                g=build_geometry(a['endpoints'],cfg)
                r=ResourceState(**{n:a[n] for n in ['q_matrix','q_ul','b_ul','u_s']})
                assert a['previous_directions'].shape==(8,3)
            source.e.check_state(g,r,cfg,{},'proposed')
            assert abs(source.e.metrics_values(g,r,cfg)['weighted_sum_rate']-d['final_score'])<1e-9
            post=[]
            for i,row in enumerate(d['rows']):
                if i:post.append(row['main_fp_steps_this_call'])
                calls.append(dict(scene=seed,after_position_iteration=i if i else '',
                    stage='after_position' if i else 'initial',
                    actual_fp_iterations=row['main_fp_steps_this_call'],
                    extrapolation_blocks=row['extrapolation_blocks'],
                    inner_tolerance_met=row['main_inner_tolerance_met'],
                    gain=row['resource_gain']))
            p=d['post_final_position_probe'];post.append(p['fp_steps'])
            calls.append(dict(scene=seed,after_position_iteration=d['iterations'],
                stage='final_position_probe_only',actual_fp_iterations=p['fp_steps'],
                extrapolation_blocks=len(p['rows']),
                inner_tolerance_met=len(p['rows'])>=2 and max(r['increment'] for r in p['rows'][-2:])<=1e-5,
                gain=p['gain']))
            out.append(dict(scene=seed,converged=d['converged'],iterations=d['iterations'],
                seconds=d['seconds'],final_wsr=d['final_score'],
                final_outer_increments=np.diff(d['history'][-3:]).tolist(),
                actual_fp_total=d['main_fp_steps']+d['candidate_fp_steps'],
                post_position_fp_median=float(np.median(post)),
                post_position_fp_mean=float(np.mean(post)),post_position_fp_range=[min(post),max(post)],
                final_position_next_inner_fp=p['fp_steps'],final_position_probe_gain=p['gain'],
                previous_o1e4_i1e6_iterations=old['iterations'],previous_o1e4_i1e6_wsr=old['final_score'],
                change_from_previous_i1e6_percent=(d['final_score']/old['final_score']-1)*100,
                loose_o1e3_i1e5_iterations=loose['iterations'],loose_o1e3_i1e5_wsr=loose['final_score'],
                additional_gain_over_loose_outer=d['final_score']-loose['final_score']))
    save(ROOT/'current_tolerance_summary.json',dict(inner_tolerance=1e-5,outer_tolerance=1e-4,
        outer_consecutive=2,rows=out,production_sources_unchanged=True))
    with (ROOT/'inner_iterations.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(calls[0]));writer.writeheader();writer.writerows(calls)
    lines=['# 当前测试：外层 1e-4、内层 1e-5', '',
        '本次已覆盖原外层 1e-4、内层 1e-6 的两个原始冻结初态测试 JSON 与最终 NPZ，并更新实验结论.md、summary.json 和 diagnostic_comparison.png。仅保留旧标量参考 replaced_o1e4_i1e6_reference.json；正式工程未改。', '',
        '场景 8003、8006，30 dBm。使用相同初态、资源 FP 外推、原 PR+、FP-aware-2。外层绝对 1e-4 连续两轮；内层接受的外推块增量 1e-5 连续两块；外层预算 120 轮，单次内层实际 FP 上限 1200。候选位置仍最多两次普通 FP。', '',
        '| 场景 | 外层轮数 | 满足停止条件 | 耗时（秒） | 最终 WSR | 相对原内层 1e-6 末值变化 |',
        '|---|---:|---|---:|---:|---:|']
    for r in out:
        lines.append(f"| {r['scene']} | {r['iterations']} | {'是' if r['converged'] else '否，预算截止'} | {r['seconds']:.2f} | {r['final_wsr']:.9f} | {r['change_from_previous_i1e6_percent']:+.5f}% |")
    lines += ['', '## 只比较外层阈值（内层固定 1e-5）', '',
        '与外层 1e-3 测试相比，两条轨迹在共同区间完全相同，因此此处额外收益可归于更晚停止；不将内层精度改动混入比较。', '',
        '| 场景 | 外层 1e-3 轮数 | 外层 1e-4 轮数 | 额外 WSR 增益 |',
        '|---|---:|---:|---:|']
    for r in out:lines.append(f"| {r['scene']} | {r['loose_o1e3_i1e5_iterations']} | {r['iterations']} | {r['additional_gain_over_loose_outer']:.9f} |")
    lines += ['', '## 更新位置后的内层实际工作量', '',
        '| 场景 | FP 次数中位数 | 范围 | 最后位置更新后再求解的 FP 次数 |',
        '|---|---:|---:|---:|']
    for r in out:lines.append(f"| {r['scene']} | {r['post_position_fp_median']:.1f} | {r['post_position_fp_range'][0]}–{r['post_position_fp_range'][1]} | {r['final_position_next_inner_fp']} |")
    lines += ['',
        '统计排除初始资源求解，包含最后位置更新后的一次独立诊断。该诊断使用最终状态副本，不改变最终 WSR、外层停止结果与上述求解耗时。实际 FP 次数包含外推与被拒试探；与外推块数、位置候选的最多 2 次 FP 分开。早期达到 1200 次上限不意味着满足内层精度。', '',
        '真实 WSR 单调性、功率、接收波束单位范数、缆绳定长及防碰撞核验通过，最终 NPZ 重算 WSR 与记录一致。最终 RCG 梯度和方向已保存。', '',
        '复现（分别指定两个场景）：', '', '```powershell',
        "& 'D:/miniconda/envs/toma_isac/python.exe' './check_accelerated_joint.py' --initial --iterations 120 --seed 8003 --inner-tolerance 1e-5 --outer-tolerance 1e-4 --final-probe", '```']
    (ROOT/'容差测试结论.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in ROOT.rglob('*') if p.is_file() and p.suffix in ('.py','.json','.npz','.md','.png','.svg','.csv')
        and p.name!='artifact_manifest.json'}
    save(ROOT/'artifact_manifest.json',manifest)
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
