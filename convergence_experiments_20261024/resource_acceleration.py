"""Monotone, power-projected squared extrapolation of the FP resource map.

Experimental only. Geometry stays fixed; evaluated true WSR safeguards every
accepted extrapolation. Rejected trial FP work counts toward the budget.
"""
import controlled_curvature as exp
from controlled_curvature import np, source, inner, outer, algo, replace, deepcopy
from controlled_curvature import build_channels, compute_performance, ROOT, save
import time, zipfile, argparse
from contextlib import ExitStack
from unittest.mock import patch
from system_model import check_resource_constraints

def vector(r):
    q = r.q_matrix.ravel()
    return np.r_[q.real,q.imag,r.q_ul]

def projected(x, template, cfg):
    r=deepcopy(template); n=r.q_matrix.size
    r.q_matrix=(x[:n]+1j*x[n:2*n]).reshape(r.q_matrix.shape)
    power=np.linalg.norm(r.q_matrix)**2
    if power>cfg.p_dl_max: r.q_matrix*=np.sqrt(cfg.p_dl_max/power)
    r.q_ul=np.clip(x[2*n:],0,np.sqrt(cfg.p_ul_max))
    return r

def accelerated_solve(ch, r, cfg, budget, max_alpha=64., target=None, tolerance=None):
    counter=0; rows=[]; start=time.perf_counter()
    def score(rr): return float(compute_performance(ch,rr,cfg).weighted_sum_rate)
    initial=score(r); value=initial; small=0
    def fp(rr):
        nonlocal counter
        result,hist=inner.solve_inner_problem(ch,rr,replace(cfg,max_inner_iter=1),verbose=False)
        counter+=1
        return result,float(hist[-1])
    while counter+3<=budget:
        before=value; x=vector(r)
        r1,f1=fp(r); r2,f2=fp(r1)
        d=vector(r1)-x; v=vector(r2)-2*vector(r1)+x
        alpha=min(max_alpha,max(1.,float(np.linalg.norm(d)/max(np.linalg.norm(v),1e-30))))
        accepted=False; attempted=alpha; backtracks=0
        while counter<budget:
            candidate=projected(x+2*alpha*d+alpha**2*v,r2,cfg)
            candidate,fc=fp(candidate)
            if fc>=f2:
                r,value=candidate,fc; accepted=True; break
            if alpha<=1.00001: break
            alpha=1.+(alpha-1.)*.5; backtracks+=1
        if not accepted: r,value=r2,f2
        check_resource_constraints(r,cfg)
        assert value>=before-1e-8
        rows.append(dict(fp_steps=counter,wsr=value,seconds=time.perf_counter()-start,
            alpha=alpha,attempted_alpha=attempted,backtracks=backtracks,
            extrapolation_accepted=accepted,increment=value-before))
        small=small+1 if tolerance is not None and value-before<=tolerance else 0
        if tolerance is not None and small>=2: break
        if target is not None and value>=target: break
    while counter<budget and (target is None or value<target) and not (tolerance is not None and small>=2):
        r,value=fp(r)
        rows.append(dict(fp_steps=counter,wsr=value,seconds=time.perf_counter()-start,
            extrapolation_accepted=False,increment=value-rows[-1]['wsr'] if rows else value-initial))
    return r,dict(initial_score=initial,final_score=value,gain=value-initial,
        seconds=time.perf_counter()-start,fp_steps=counter,rows=rows)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--budget',type=int,default=1500)
    p.add_argument('--powers',nargs='+',default=['0.1','1','10'])
    p.add_argument('--seed',type=int,default=8003)
    args=p.parse_args(); hashes=exp.production_hashes()
    with zipfile.ZipFile(source.ZIP) as z,ExitStack() as stack:
        for mod in (inner,outer,algo):stack.enter_context(patch.object(mod,'relative_change',source.e.abs_change))
        for power in args.powers:
            cfg,g,r=source.load(z,power,args.seed); ch=build_channels(g,cfg)
            initial=source.e.metrics_values(g,r,cfg)['weighted_sum_rate']
            tick=time.perf_counter()
            with patch.object(inner,'relative_change',lambda a,b:float('inf')):
                plain,hh=inner.solve_inner_problem(ch,deepcopy(r),replace(cfg,max_inner_iter=args.budget),verbose=False)
            plain_result=dict(initial_score=initial,final_score=hh[-1],gain=hh[-1]-initial,
                seconds=time.perf_counter()-tick,fp_steps=len(hh)-1,history=list(map(float,hh)))
            rr,result=accelerated_solve(ch,deepcopy(r),cfg,args.budget)
            source.e.check_state(g,rr,cfg,{},'proposed')
            suffix='' if args.seed==8003 else f'_s{args.seed}'
            save(ROOT/f'resource_acceleration_p{power}_b{args.budget}{suffix}.json',dict(
                power_W=float(power),scene=args.seed,plain=plain_result,accelerated=result,
                note='Fixed saved final geometry; equal actual FP iteration budget. No outer convergence claim.'))
            print({k:v for k,v in dict(power=power,seed=args.seed,plain_gain=plain_result['gain'],
                accelerated_gain=result['gain'],plain_s=plain_result['seconds'],
                accelerated_s=result['seconds'],accepted=sum(x['extrapolation_accepted'] for x in result['rows'])).items()},flush=True)
    assert hashes==exp.production_hashes()

if __name__=='__main__':main()
