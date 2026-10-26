"""Deterministic, feasible planar ToMA endpoints and regular MA/UPA starts."""
from copy import deepcopy
import numpy as np


def regular_toma_geometry(cfg):
    """Uniform endpoint angles on the x=0 great circle; all original constraints.

    For M=4, phase pi/4 gives four square vertices on a circle of radius Lc.
    Antennas retain p_mn=(n/Nc)*c_m. Optimization is NOT restricted to this plane.
    """
    from geometry import build_geometry
    theta = np.pi/4 + 2*np.pi*np.arange(cfg.m_uav)/cfg.m_uav
    endpoints = np.zeros((cfg.m_uav, 3), dtype=float)
    endpoints[:, 1] = cfg.cable_length*np.cos(theta)
    endpoints[:, 2] = cfg.cable_length*np.sin(theta)
    return build_geometry(endpoints, cfg, check_constraints=True)


def make_initials(cfg, seed=2042):
    from channels import build_channels
    from initialization import initialize_resources
    from baselines import build_fpa_upa_geometry
    toma = regular_toma_geometry(cfg)
    upa = build_fpa_upa_geometry(cfg)
    rt = initialize_resources(toma, build_channels(toma,cfg), cfg, np.random.default_rng(seed))
    ru = initialize_resources(upa, build_channels(upa,cfg), cfg, np.random.default_rng(seed))
    return {'proposed': (deepcopy(toma),deepcopy(rt)),
            'b2': (deepcopy(toma),deepcopy(rt)),
            'b6': (deepcopy(upa),deepcopy(ru)),
            'b3': (deepcopy(upa),deepcopy(ru))}


def save_initials(folder, initials):
    folder.mkdir(parents=True,exist_ok=True)
    for scheme,(g,r) in initials.items():
        np.savez_compressed(folder/f'{scheme}.npz', endpoints=g.endpoints,
            tx_positions=g.tx_positions, rx_positions=g.rx_positions,
            q_matrix=r.q_matrix, q_ul=r.q_ul, b_ul=r.b_ul, u_s=r.u_s)


def plot_initials(initials,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(10,4.8))
    g,_=initials['proposed'];ax=axs[0]
    ax.scatter([0],[0],c='black',marker='x',s=65,label='Cable anchor')
    for m,c in enumerate(g.endpoints):
        tx=m<len(g.endpoints)//2;color='#dc3333' if tx else '#247ac1'
        ax.plot([0,c[1]],[0,c[2]],c=color,alpha=.55)
        ax.scatter([.5*c[1],c[1]],[.5*c[2],c[2]],c=color,s=45,
                   label=('Tx' if tx else 'Rx') if m in (0,len(g.endpoints)//2) else None)
        ax.annotate(f'c{m+1}',(c[1],c[2]),xytext=(5,5),textcoords='offset points')
    ax.set_title('ToMA: uniform endpoints in x=0 plane')
    ax.set_xlabel('y (m)');ax.set_ylabel('z (m)');ax.legend(fontsize=8)
    g,_=initials['b6'];ax=axs[1]
    ax.scatter(g.tx_positions[:,1],g.tx_positions[:,2],marker='s',s=110,
               facecolors='none',edgecolors='#dc3333',label='Tx: x=-0.5 m')
    ax.scatter(g.rx_positions[:,1],g.rx_positions[:,2],marker='x',s=55,
               c='#247ac1',label='Rx: x=+0.5 m')
    ax.set_title('MA and UPA: identical regular initial arrays')
    ax.set_xlabel('y (m)');ax.set_ylabel('z (m)');ax.legend(fontsize=8)
    for ax in axs:ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.25);ax.margins(.2)
    fig.text(.5,.015,'Initial geometry only. Original ToMA sphere / MA region constraints remain active during optimization.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(path,dpi=180);plt.close(fig)
