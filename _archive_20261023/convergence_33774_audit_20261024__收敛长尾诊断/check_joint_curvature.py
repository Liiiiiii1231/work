"""Joint tangent Hessian of a fixed-two-FP candidate map at one saved state."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import zipfile,json,time
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch
import numpy as np
import check_larger_step as test
import inner_solver as inner,outer_solver as outer
from channel_model import build_channels
from system_model import build_geometry

def main():
    with zipfile.ZipFile(test.ZIP) as z:cfg,g,r=test.load(z,'1',8003)
    with patch.object(inner,'relative_change',test.e.abs_change):
        r,history=inner.solve_inner_problem(build_channels(g,cfg),r,cfg,verbose=False)
    bases=[]
    for i in range(cfg.m_uav):
        grad=outer.project_to_tangent(g.endpoints[i],outer.central_difference_gradient(g.endpoints,i,r,replace(cfg,finite_diff_delta=1e-7)),cfg)
        u=grad/np.linalg.norm(grad);v=np.cross(g.endpoints[i],u);v=v/np.linalg.norm(v)
        bases.append(np.array([u,v]))
    def value(x):
        ep=g.endpoints.copy()
        for i in range(cfg.m_uav):ep[i]=outer.sphere_retraction(ep[i],x[2*i:2*i+2]@bases[i],1.,cfg)
        ch=build_channels(build_geometry(ep,cfg,check_constraints=False),cfg)
        rr,h=inner.solve_inner_problem(ch,deepcopy(r),replace(cfg,max_inner_iter=2),verbose=False)
        return h[-1]
    rows=[];hessians=[]
    with patch.object(inner,'relative_change',lambda a,b:float('inf')):
        for h in (1e-6,3e-6):
            tick=time.perf_counter();n=2*cfg.m_uav;zero=np.zeros(n);f0=value(zero)
            gradient=np.zeros(n);hes=np.zeros((n,n))
            for i in range(n):
                x=np.zeros(n);x[i]=h
                fp=value(x);fm=value(-x);gradient[i]=(fp-fm)/(2*h);hes[i,i]=(fp-2*f0+fm)/(h*h)
                for j in range(i):
                    y=np.zeros(n);y[j]=h
                    hes[i,j]=hes[j,i]=(value(x+y)-value(x-y)-value(-x+y)+value(-x-y))/(4*h*h)
            block=np.zeros_like(hes)
            for i in range(cfg.m_uav):block[2*i:2*i+2,2*i:2*i+2]=hes[2*i:2*i+2,2*i:2*i+2]
            diag=np.diag(np.diag(hes));eigen=np.linalg.eigvalsh(hes)
            row=dict(delta=h,seconds=time.perf_counter()-tick,gradient=gradient.tolist(),hessian=hes.tolist(),
                eigenvalues=eigen.tolist(),block_diagonal_eigenvalues=np.linalg.eigvalsh(block).tolist(),
                off_endpoint_block_norm_ratio=float(np.linalg.norm(hes-block)/np.linalg.norm(hes)),
                off_coordinate_diagonal_norm_ratio=float(np.linalg.norm(hes-diag)/np.linalg.norm(hes)))
            rows.append(row);hessians.append(hes)
            print({k:v for k,v in row.items() if k not in ('hessian','gradient','block_diagonal_eigenvalues')},flush=True)
    test.e.json_write(test.OUT/'joint_tangent_curvature.json',dict(rows=rows,
        hessian_relative_difference=float(np.linalg.norm(hessians[0]-hessians[1])/np.linalg.norm(hessians[0])),
        note='Local tangent curvature of exactly-two-FP candidate map, saved scene8003 30dBm final state; neither exact fully optimized value-function Hessian nor global convergence proof.'))

if __name__=='__main__':main()
