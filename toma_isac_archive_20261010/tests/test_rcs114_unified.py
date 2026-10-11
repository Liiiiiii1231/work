"""新版正式配置：车辆 RCS=10 m²，BS=-114dBm，DL=-70dBm。"""
from pathlib import Path
import numpy as np
import yaml
import pytest
from config import load_config
from channels import sensing_path_gains, communication_path_gains

ROOT=Path(__file__).resolve().parents[1]
CONFIG_DIR=ROOT/'configs'
BS_NOISE_W=10.0**((-114.0-30.0)/10.0)

@pytest.mark.parametrize('yaml_file',sorted(p.name for p in CONFIG_DIR.glob('*.yaml') if p.name!='stage5_multiuser.yaml'))
def test_all_official_configs(yaml_file):
    path=CONFIG_DIR/yaml_file
    cfg=load_config(path)
    with path.open(encoding='utf-8') as f: raw=yaml.safe_load(f)
    assert 'xi_s' not in raw['channel']
    assert raw['channel']['rcs_m2']==[10.0]*cfg.l_target
    np.testing.assert_allclose(cfg.xi_s, np.full(cfg.l_target,np.sqrt(4*np.pi*10.0)/cfg.wavelength),rtol=1e-13)
    np.testing.assert_allclose(cfg.xi_s.imag,0)
    np.testing.assert_allclose(cfg.sigma_bs2,BS_NOISE_W,rtol=1e-13)
    np.testing.assert_array_equal(cfg.sigma_dl2,np.full(cfg.k_dl,1e-10))
    np.testing.assert_array_equal(cfg.fading_s,np.ones(cfg.l_target))
    _,_,alpha=sensing_path_gains(cfg)
    r=np.linalg.norm(cfg.q_s,axis=1)
    np.testing.assert_allclose(np.abs(alpha)**2,cfg.wavelength**2*10/((4*np.pi)**3*r**4),rtol=1e-12)
    dl,ul=communication_path_gains(cfg)
    assert np.all(np.isfinite(dl)) and np.all(np.isfinite(ul))

def test_legacy_stage5_not_modified():
    cfg=load_config(CONFIG_DIR/'stage5_multiuser.yaml')
    np.testing.assert_array_equal(cfg.xi_s,np.ones(cfg.l_target))
    assert cfg.sigma_bs2==1e-10

def test_tradeoff_endpoints_work():
    pytest.importorskip("cvxpy")
    from experiments.run_stage10_tradeoff import load_tradeoff_config,make_group_weights
    cfg=load_config(CONFIG_DIR/'stage10_tradeoff.yaml')
    tradeoff=load_tradeoff_config(CONFIG_DIR/'stage10_tradeoff.yaml')
    assert 0.0 in tradeoff.omega_comm
    for omega in (0,1e-10,0.5,1):
        a,b,c=make_group_weights(cfg,omega)
        assert np.isclose(np.sum(a)+np.sum(b)+np.sum(c),1)
