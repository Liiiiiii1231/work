"""Fast integration test with the real channel/performance evaluator but no FP optimization.
Run in a project installation: python -m pytest tests/test_pool_refinement.py -q
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from baselines import build_fpa_upa_geometry
from channels import build_channels
from initialization import initialize_resources
from metrics import compute_performance
from state import GeometryState, ResourceState


def load_runner():
    p = Path(__file__).resolve().parents[1] / "experiments" / "run_stage10_pool_refinement.py"
    spec = importlib.util.spec_from_file_location("stage10_pool_runner", p)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_npz_roundtrip_and_weighted_ranking(tmp_path, monkeypatch):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config import load_config
    m = load_runner()
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "stage10_tradeoff.yaml")
    geo = build_fpa_upa_geometry(cfg)
    ch = build_channels(geo, cfg)
    resources = initialize_resources(geo,ch,cfg,np.random.default_rng(2026))
    perf = compute_performance(ch,resources,cfg)
    fake = SimpleNamespace(geometry=geo,resources=resources,performance=perf,
                           converged=True, iterations=4)
    p = tmp_path / "candidate.npz"
    m.save_state(p, fake)
    g,r,rates = m.read_state(p)
    assert g.tx_positions.shape == geo.tx_positions.shape
    assert r.q_matrix.shape == resources.q_matrix.shape
    assert np.isclose(m.verify_state(p,cfg),perf.weighted_sum_rate,rtol=1e-6)
    assert np.isclose(m.weighted_rates(*rates,cfg),perf.weighted_sum_rate,rtol=1e-6)


def test_cold_and_pool_mock(tmp_path, monkeypatch):
    m = load_runner()
    from config import load_config
    from stage8 import load_stage8_experiment_config, make_scene_config
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "stage10_tradeoff.yaml"
    cfg = load_config(cfg_path)
    e = load_stage8_experiment_config(cfg_path)
    scene = make_scene_config(cfg, scene_seed=8001, position_jitter=e.position_jitter,
                              randomize_fading=False, fading_model=e.fading_model)
    calls = []

    def fake_solver(scheme,cfg,exp,seed,warm=None):
        calls.append(warm is not None)
        if warm is not None:
            geom = warm[0]
            r = warm[1]
        else:
            geom = build_fpa_upa_geometry(cfg)
            ch = build_channels(geom,cfg)
            r = initialize_resources(geom,ch,cfg,np.random.default_rng(seed))
        ch = build_channels(geom,cfg)
        perf = compute_performance(ch,r,cfg)
        return SimpleNamespace(geometry=geom,resources=r,performance=perf,
                               converged=True,iterations=4)

    monkeypatch.setattr(m,"run_solver",fake_solver)
    w = (0.2,0.3)
    m.run_cold(tmp_path,scene,e,w,("b6",),2,8001)
    m.audit_candidates(tmp_path,scene,w,("b6",),1,e)
    m.pareto_audit(tmp_path)
    assert (tmp_path/"stage10_tradeoff.csv").exists()
    assert (tmp_path/"candidate_audit.csv").exists()
    assert calls.count(False)==4
    assert calls.count(True)==2
    with (tmp_path/"stage10_tradeoff.csv").open(encoding="utf-8-sig") as f:
        import csv
        rows = list(csv.DictReader(f))
    assert len(rows)==2
    assert all(row["converged"]=="True" for row in rows)
    # All optimization jobs should be skipped when resumed.
    m.run_cold(tmp_path,scene,e,w,("b6",),2,8001)
    m.audit_candidates(tmp_path,scene,w,("b6",),1,e)
    assert len(calls)==6
