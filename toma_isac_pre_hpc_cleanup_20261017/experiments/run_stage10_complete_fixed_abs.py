"""Supplement only Fixed-ToMA/UPA, reuse completed Proposed/MA, plot 4 schemes.

Place in experiments/ and run from project root. No existing solver/config or
result is overwritten. No position optimization and no cross-weight warm start.
Inner stopping is absolute WSR change <= 1e-4, once, exactly as the top10 run.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch

import numpy as np
import yaml

LABELS = {"b2":"Fixed-ToMA + FP", "b3":"FPA-UPA + FP", "b6":"MA + FP",
          "proposed":"FP-aware RCG-ToMA + FP (Proposed)"}
PLOT_FIELDS = ["omega_comm", "omega_sensing", "scheme", "scheme_label", "sum_rate_dl", "sum_rate_ul",
               "sum_rate_comm", "sum_rate_sensing", "weighted_sum_rate", "best_seed", "iterations", "runtime_s", "n_starts", "converged"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    tmp = path.with_suffix(path.suffix+".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def write_json(path, value):
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def truth(value):
    return str(value).lower() in ("true", "1")


def geometry_digest(geometry):
    # Identical to run_stage10_top10_abs.py, not a hash of endpoint coordinates alone.
    values = np.concatenate([geometry.tx_positions.ravel(), geometry.rx_positions.ravel()]).astype("<f8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def nondominated(rows):
    xy = np.array([[float(r["sum_rate_comm"]), float(r["sum_rate_sensing"])] for r in rows])
    keep = [i for i,p in enumerate(xy) if not any(np.all(q>=p-1e-10) and np.any(q>p+1e-10) for q in xy)]
    return sorted((rows[i] for i in keep), key=lambda r:float(r["sum_rate_comm"]))


def plot(rows, folder, diagnostic=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10})
    fig, axs = plt.subplots(1, 2, figsize=(13.5,5.8), sharex=True, sharey=True)
    styles = [("b2","#247ac1","o"),("b3","#ef8415","s"),("b6","#279b3b","^"),("proposed","#df2228","D")]
    filtered = []
    for scheme, color, marker in styles:
        data = sorted((r for r in rows if r["scheme"]==scheme), key=lambda r:float(r["omega_comm"]))
        if not data:
            continue
        front = nondominated(data)
        filtered.extend(front)
        x = [float(r["sum_rate_comm"]) for r in data]
        y = [float(r["sum_rate_sensing"]) for r in data]
        axs[0].plot(x,y,color=color,marker=marker,lw=1.7,ms=4.5,mec="white",mew=.5,label=LABELS[scheme])
        axs[1].scatter(x,y,color=color,marker=marker,s=22,alpha=.25)
        axs[1].plot([float(r["sum_rate_comm"]) for r in front], [float(r["sum_rate_sensing"]) for r in front],
                    color=color,marker=marker,lw=1.7,ms=4.5,mec="white",mew=.5,label=LABELS[scheme])
    for ax in axs:
        ax.grid(alpha=.22)
        ax.set_xlabel("Sum communication rate (bit/s/Hz)")
    axs[0].set_ylabel("Sum sensing rate (bit/s/Hz)")
    axs[0].set_title("(a) All points connected in weight order",fontsize=11)
    axs[1].set_title("(b) Nondominated sampled points highlighted",fontsize=11)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.99),ncol=2,fontsize=9,frameon=True)
    fig.text(.07,.047,"Faint markers in (b) retain all original points. Filtering is applied separately and identically to every scheme.",fontsize=8.5)
    fig.text(.07,.015,"No smoothing or new intermediate solutions. Connecting segments are visual guides, not a verified global Pareto boundary.",fontsize=8.5)
    fig.subplots_adjust(left=.065,right=.985,bottom=.17,top=.79,wspace=.12)
    folder.mkdir(exist_ok=True)
    stem = "communication_sensing_tradeoff_two_panels"+("_diagnostic" if diagnostic else "")
    for ext in ("png","pdf","svg"):
        fig.savefig(folder/f"{stem}.{ext}",dpi=240,facecolor="white")
    plt.close(fig)
    write_csv(folder.parent/"nondominated_sample_points.csv", filtered, PLOT_FIELDS)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root")
    ap.add_argument("--base-run", default="results/stage10/top10_abs1e4_scene8001")
    ap.add_argument("--source", default="results/stage10/top10_abs1e4_scene8001_b6_09_continued")
    ap.add_argument("--output", default="results/stage10/four_schemes_top10_abs1e4_scene8001")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--continue-nonconverged", action="store_true",
                    help="With --resume, continue capped fixed-array candidates from their saved resources.")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--max-jobs", type=int, default=None, help="Optional cap on jobs for this invocation.")
    ap.add_argument("--smoke", action="store_true", help="Separate validation run: one weight and one seed per fixed scheme.")
    args = ap.parse_args()
    if args.max_jobs is not None and args.max_jobs<1:
        ap.error("max-jobs must be positive.")
    if args.continue_nonconverged and not args.resume:
        ap.error("Use --resume with --continue-nonconverged.")
    root = Path(args.project_root).resolve() if args.project_root else Path.cwd().resolve()
    if not (root/"baselines.py").is_file():
        root = Path(__file__).resolve().parent.parent
    if not (root/"baselines.py").is_file():
        raise RuntimeError("Run from project root or specify --project-root.")
    sys.path.insert(0,str(root))
    base_run, source, out = ((root/x).resolve() for x in (args.base_run,args.source,args.output))
    if args.smoke:
        out = out.with_name(out.name+"_smoke")
    if any(out==p or p in out.parents for p in (source,base_run)):
        raise RuntimeError("Output must be outside both original result directories.")
    if args.plot_only:
        manifest = json.loads((out/"completion.json").read_text(encoding="utf-8"))
        if not manifest["all_points_present"] or manifest["smoke"]:
            raise RuntimeError("No complete formal four-scheme table yet.")
        plot(read_csv(out/"stage10_tradeoff.csv"),out/"figures",manifest["diagnostic"])
        print(f"Replotted: {out}")
        return

    meta = json.loads((base_run/"run_metadata.json").read_text(encoding="utf-8"))
    fp, raw = meta["fingerprint"],meta["config"]
    if fp.get("stopping")!="absolute_wsr_change" or fp.get("inner_consecutive")!=1:
        raise RuntimeError("Expected top10 absolute stopping metadata.")
    if raw["algorithm"]["tol_inner"]!=1e-4 or raw["algorithm"]["tol_outer"]!=1e-4:
        raise RuntimeError("Expected original tolerances 1e-4.")
    if raw["experiment"].get("fading",{}).get("enabled",False):
        raise RuntimeError("This entry expects the original deterministic LoS scene.")
    code_names = ("baselines.py","channels.py","config.py","geometry.py","initialization.py","inner_solver.py","metrics.py","state.py","utils.py","stage8.py")
    for name in code_names:
        if sha(root/name)!=fp["code_sha256"][name]:
            raise RuntimeError(f"Code differs from source experiment: {name}.")
    source_rows = read_csv(source/"stage10_tradeoff.csv")
    starts_source = read_csv(source/"all_starts.csv")
    weights = [float(w) for w in fp["weights"]]
    seeds = [int(s) for s in fp["seeds"]["proposed"]]
    upa_seed = int(raw["experiment"]["fixed_upa_seed"])
    if len(seeds)!=10 or len(weights)!=20 or int(fp["scene_seed"])!=8001:
        raise RuntimeError("Expected scene8001, 10 selected ToMA seeds, 20 weights.")
    source_map = {(r["scheme"],float(r["omega_comm"])):r for r in source_rows}
    if len(source_rows)!=40 or len(source_map)!=40:
        raise RuntimeError("Completed source must contain exactly 40 unique Proposed/MA points.")
    for w in weights:
        for scheme in ("b6","proposed"):
            r = source_map[scheme,w]
            if not truth(r["converged"]):
                raise RuntimeError("Source includes nonconverged selected point.")
            matches = [s for s in starts_source if s["scheme"]==scheme and float(s["omega_comm"])==w and int(s["seed"])==int(r["best_seed"])]
            if len(matches)!=1 or not truth(matches[0]["converged"]) or not np.isclose(float(matches[0]["weighted_sum_rate"]),float(r["weighted_sum_rate"]),rtol=1e-10,atol=1e-10):
                raise RuntimeError("Source selected point does not match its candidate record.")
    old_map = {(r["scheme"],float(r["omega_comm"])):r for r in read_csv(base_run/"stage10_tradeoff.csv")}
    for key,row in source_map.items():
        if key!=("b6",.9) and row!=old_map[key]:
            raise RuntimeError("Source differs beyond the authorized B6 weight0.9 continuation.")

    identity = dict(metadata_sha256=sha(base_run/"run_metadata.json"),source_points_sha256=sha(source/"stage10_tradeoff.csv"),
                    source_starts_sha256=sha(source/"all_starts.csv"),runner_sha256=sha(Path(__file__).resolve()),
                    code_sha256={n:sha(root/n) for n in code_names},smoke=args.smoke)
    if out.exists():
        if not args.resume:
            raise RuntimeError(f"Output exists: {out}; use --resume.")
        if json.loads((out/"run_metadata.json").read_text(encoding="utf-8"))["identity"]!=identity:
            raise RuntimeError("Code or original inputs changed; refusing mixed-run resume.")
    else:
        if args.resume:
            raise RuntimeError("No previous output to resume.")
        out.mkdir(parents=True)
        for folder in ("states","records"):
            (out/folder).mkdir()
        write_json(out/"run_metadata.json",dict(identity=identity,config=raw,selected_seeds=seeds,
                                               scene_seed=8001,weights=weights,stopping="absolute",tol_inner=1e-4,
                                               inner_consecutive=1,source=str(source)))

    import baselines
    import inner_solver as inner
    from channels import build_channels
    from config import load_config
    from geometry import build_geometry
    from initialization import generate_feasible_endpoints,initialize_resources
    from metrics import compute_performance
    from stage8 import make_scene_config
    from state import ResourceState
    from utils import check_resource_constraints
    with TemporaryDirectory(prefix="stage10_fixed_config_") as tmp:
        p=Path(tmp)/"original.yaml";p.write_text(yaml.safe_dump(raw,allow_unicode=True),encoding="utf-8")
        base=load_config(p)
    scene=make_scene_config(base,8001,np.asarray(raw["experiment"]["position_jitter"],float),randomize_fading=False)
    initials={}
    for seed in seeds:
        rng=np.random.default_rng(seed)
        g=build_geometry(generate_feasible_endpoints(scene,rng),scene)
        if geometry_digest(g)!=fp["initial_geometry_sha256"][f"proposed_{seed}"]:
            raise RuntimeError(f"Initial layout differs from Proposed seed {seed}.")
        c=build_channels(g,scene)
        initials["b2",seed]=(g,initialize_resources(g,c,scene,rng))
    g=baselines.build_fpa_upa_geometry(scene)
    c=build_channels(g,scene)
    initials["b3",upa_seed]=(g,initialize_resources(g,c,scene,np.random.default_rng(upa_seed)))
    jobs=[(w,sc,s) for w in weights for sc,ss in (("b2",seeds),("b3",[upa_seed])) for s in ss]
    if args.smoke:
        jobs=[(.1,"b2",seeds[0]),(.1,"b3",upa_seed)]
    print(f"Scene8001; Fixed-ToMA10 starts + fixed UPA1 start; jobs={len(jobs)}; inner absolute tolerance=1e-4; max inner={scene.max_inner_iter}",flush=True)
    print("Reusing all 40 completed Proposed/MA points; no position optimization.",flush=True)

    def record_path(w,sc,s):
        return out/"records"/f"omega_{w:g}_{sc}_seed_{s}.json"

    def collect():
        return [json.loads(record_path(*job).read_text(encoding="utf-8")) for job in jobs if record_path(*job).is_file()]

    def summarize():
        records=collect()
        rows=[r["row"] for r in records]
        start_fields=list(dict.fromkeys(k for row in rows for k in row)) or ["omega_comm","scheme","seed","converged"]
        write_csv(out/"fixed_all_starts.csv",rows,start_fields)
        write_csv(out/"fixed_nonconverged_starts.csv",[r for r in rows if not truth(r["converged"])],start_fields)
        missing,higher,chosen=[],[],[]
        for w in weights:
            for sc,ss in (("b2",seeds),("b3",[upa_seed])):
                pool=[r for r in rows if r["scheme"]==sc and float(r["omega_comm"])==w]
                valid=[r for r in pool if truth(r["converged"])]
                if not valid:
                    missing.append(dict(omega_comm=w,scheme=sc,completed_candidates=len(pool),expected_candidates=len(ss)))
                    continue
                best=max(valid,key=lambda r:float(r["weighted_sum_rate"]))
                for r in pool:
                    if not truth(r["converged"]) and float(r["weighted_sum_rate"])>float(best["weighted_sum_rate"])+1e-9:
                        higher.append(r)
                row={k:best[k] for k in ("sum_rate_dl","sum_rate_ul","sum_rate_comm","sum_rate_sensing","weighted_sum_rate","iterations","runtime_s","converged")}
                row.update(omega_comm=w,omega_sensing=1-w,scheme=sc,scheme_label=LABELS[sc],best_seed=best["seed"],n_starts=len(ss))
                chosen.append(row)
        merged=sorted([dict(r) for r in source_rows]+chosen,key=lambda r:(float(r["omega_comm"]),list(LABELS).index(r["scheme"])))
        write_csv(out/"stage10_tradeoff.csv",merged,PLOT_FIELDS)
        write_csv(out/"missing_points.csv",missing,["omega_comm","scheme","completed_candidates","expected_candidates"])
        write_csv(out/"higher_unconverged_candidates.csv",higher,start_fields)
        all_present=len(rows)==len(jobs) and not missing and not args.smoke
        completion=dict(smoke=args.smoke,expected_jobs=len(jobs),completed_jobs=len(rows),
                        converged_jobs=sum(truth(r["converged"]) for r in rows),all_points_present=all_present,
                        plot_points=len(merged),diagnostic=bool(higher),higher_unconverged_candidates=len(higher),missing_points=len(missing))
        write_json(out/"completion.json",completion)
        return completion,merged

    done=0
    for w,sc,seed in jobs:
        rp=record_path(w,sc,seed)
        previous=json.loads(rp.read_text(encoding="utf-8")) if rp.is_file() else None
        if previous and (truth(previous["row"]["converged"]) or not args.continue_nonconverged):
            continue
        if args.max_jobs is not None and done>=args.max_jobs:
            break
        cfg=replace(scene,weight_dl=np.full(scene.k_dl,w/(2*scene.k_dl)),weight_ul=np.full(scene.j_ul,w/(2*scene.j_ul)),weight_s=np.full(scene.l_target,(1-w)/scene.l_target))
        cfg.validate()
        g,initial=initials[sc,seed]
        initial_wsr=float(compute_performance(build_channels(g,cfg),initial,cfg).weighted_sum_rate)
        resources=deepcopy(initial)
        old_history=[]
        if previous:
            state_path=out/previous["row"]["state_file"]
            if sha(state_path)!=previous["row"]["state_sha256"]:
                raise RuntimeError("Resume state checksum mismatch.")
            with np.load(state_path,allow_pickle=False) as a:
                if not np.array_equal(g.tx_positions,a["tx_positions"]) or not np.array_equal(g.rx_positions,a["rx_positions"]):
                    raise RuntimeError("Fixed geometry changed in checkpoint.")
                resources=ResourceState(q_matrix=a["q_matrix"].copy(),q_ul=a["q_ul"].copy(),b_ul=a["b_ul"].copy(),u_s=a["u_s"].copy())
                old_history=a["wsr_history"].tolist()
        print(f"[{len(collect())+int(previous is None)}/{len(jobs)}] {LABELS[sc]}, weight={w:g}, seed={seed}"+(" (continue)" if previous else ""),flush=True)
        tick=perf_counter()
        with patch.object(inner,"relative_change",lambda new,old:abs(float(new)-float(old))),patch.object(baselines,"relative_change",lambda new,old:abs(float(new)-float(old))):
            if sc=="b2":
                result=baselines.run_fixed_toma_fp(cfg,rng=np.random.default_rng(seed),initial_endpoints=g.endpoints.copy(),initial_resources=resources)
            else:
                result=baselines.run_fpa_upa_fp(cfg,rng=np.random.default_rng(seed),initial_resources=resources)
        elapsed=perf_counter()-tick
        check_resource_constraints(result.resources,cfg)
        if not np.array_equal(g.tx_positions,result.geometry.tx_positions) or not np.array_equal(g.rx_positions,result.geometry.rx_positions):
            raise RuntimeError("A fixed-array baseline changed its antenna positions.")
        history=list(map(float,result.wsr_history))
        expected_start=old_history[-1] if previous else initial_wsr
        if not np.isclose(history[0],expected_start,rtol=1e-9,atol=1e-9):
            raise RuntimeError("Resource initialization/resume did not reproduce.")
        if any(b<a-1e-8*max(1,abs(a)) for a,b in zip(history,history[1:])):
            raise RuntimeError("Material WSR decrease in FP history.")
        last_change=abs(history[-1]-history[-2]) if len(history)>1 else float("inf")
        converged=bool(last_change<=cfg.tol_inner)
        if converged!=bool(result.converged):
            raise RuntimeError("Baseline convergence flag disagrees with absolute stopping.")
        perf=compute_performance(result.channels,result.resources,cfg)
        dl,ul,sensing=map(float,(np.sum(perf.rate_dl),np.sum(perf.rate_ul),np.sum(perf.rate_s)))
        value=float(perf.weighted_sum_rate)
        if not np.isclose(value,w*(dl+ul)/10+(1-w)*sensing/5,rtol=1e-9,atol=1e-9):
            raise RuntimeError("Original WSR definition did not reproduce.")
        attempt=previous["attempt"]+1 if previous else 1
        full_history=old_history+history[1:] if previous else history
        state_name=f"states/omega_{w:g}_{sc}_seed_{seed}_attempt_{attempt:03d}.npz"
        tmp=(out/state_name).with_suffix(".tmp")
        with tmp.open("wb") as f:
            np.savez_compressed(f,omega_comm=w,scheme=sc,seed=seed,endpoints=g.endpoints,tx_positions=g.tx_positions,rx_positions=g.rx_positions,
                                q_matrix=result.resources.q_matrix,q_ul=result.resources.q_ul,b_ul=result.resources.b_ul,u_s=result.resources.u_s,
                                wsr_history=np.asarray(full_history),converged=converged,initial_endpoints=g.endpoints)
        os.replace(tmp,out/state_name)
        row=dict(omega_comm=w,scheme=sc,seed=seed,sum_rate_dl=dl,sum_rate_ul=ul,sum_rate_comm=dl+ul,sum_rate_sensing=sensing,weighted_sum_rate=value,
                 iterations=len(full_history)-1,converged=converged,stop_reason="inner_tolerance" if converged else "max_inner_iterations",
                 runtime_s=elapsed+(float(previous["row"]["runtime_s"]) if previous else 0),state_file=state_name,state_sha256=sha(out/state_name),
                 last_abs_change=last_change,initial_wsr=initial_wsr,initial_geometry_sha256=geometry_digest(g))
        write_json(rp,dict(attempt=attempt,row=row))
        done+=1
        summarize()
        print(f"  WSR={value:.9f}, inner={len(full_history)-1}, converged={converged}, time={elapsed:.2f}s",flush=True)
    completion,merged=summarize()
    if completion["all_points_present"]:
        plot(merged,out/"figures",completion["diagnostic"])
    print(json.dumps(completion,indent=2),flush=True)
    print(f"Output: {out}",flush=True)
    if not completion["all_points_present"]:
        print("No complete formal figure yet. --resume finishes missing jobs; --resume --continue-nonconverged also continues capped candidates.",flush=True)
    elif completion["diagnostic"]:
        print("Figure is diagnostic: a higher nonconverged candidate remains. Use --resume --continue-nonconverged.",flush=True)


if __name__=="__main__":
    main()
