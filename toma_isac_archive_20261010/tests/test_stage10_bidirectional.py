"""Fast solver-free tests of Stage 10 cold/forward/backward orchestration."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_stage10_bidirectional import Candidate, consider, scan, SCHEMES


def synthetic_result(s, c, r, *, conv=True, iters=2):
    return SimpleNamespace(
        resources=SimpleNamespace(tag=f"{s}-{c}-{r}"),
        geometry=SimpleNamespace(endpoints=np.ones((4, 3))),
        performance=SimpleNamespace(
            weighted_sum_rate=c,
            rate_dl=np.array([r / 2]), rate_ul=np.array([r / 2]),
            rate_s=np.array([c]),
        ),
        converged=conv,
        iterations=iters,
        outer_iterations=iters,
    )


def test_picks_best_converged_per_scheme_and_weight(tmp_path):
    weights = (0.005, 0.3, 0.9)
    events = []
    best_wsr_cold = {(s, w): (1 if w == 0.3 else 5) for s in SCHEMES for w in weights}

    def cfg(w):
        return SimpleNamespace(omega=w)

    def cold(s, config, omega):
        return [Candidate(synthetic_result(s, best_wsr_cold[s, omega], omega),
                          s, omega, "cold", 2026, 1, 1.0),
                Candidate(synthetic_result(s, 99, omega, conv=False),
                          s, omega, "cold", 2027, 2, 1.0)]

    def warm(s, config, donor):
        # At omega .3, source from .005 is better (5 -> 8);
        # backward from .9 cannot beat it (6). Other weights stay at 5.
        events.append((s, config.omega, donor.omega, donor.scheme))
        if config.omega == 0.3:
            value = 8.0 if donor.omega == 0.005 else 6.0
        else:
            value = 4.0
        return synthetic_result(s, value, config.omega)

    output = tmp_path / "results"
    best = scan(weights, SCHEMES, cfg, cold, warm,
                lambda donor, cfg: donor.wsr - cfg.omega,
                lambda r, cfg: None, output)

    assert len(best) == 12
    assert len(events) == 16  # 4 schemes * 2 warm moves per direction
    assert all(s == donor_s for s, _, _, donor_s in events)
    for s in SCHEMES:
        assert best[s, 0.3].source_type == 'forward'
        assert best[s, 0.3].wsr == 8.0
        assert best[s, 0.005].wsr == 5.0
        assert best[s, 0.9].wsr == 5.0
    with (output / 'stage10_tradeoff.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 12
    assert all(x['converged'] == 'True' for x in rows)
    assert all(float(x['weighted_sum_rate']) < 99 for x in rows)
    with (output / 'candidate_attempts.csv').open(encoding='utf-8', newline='') as f:
        attempts = list(csv.DictReader(f))
    assert len(attempts) == 40  # 24 cold + 16 warm
    assert sum(x['source_type'] == 'forward' for x in attempts) == 8
    assert sum(x['source_type'] == 'backward' for x in attempts) == 8
    assert not list(csv.DictReader((output / 'stage10_failures.csv').open(encoding='utf-8')))


def test_nonconverged_cold_is_rescued_by_warm_start(tmp_path):
    weights = (0.005, 0.3)
    def cold(s, cfg, w):
        return [Candidate(synthetic_result(s, 2.0, w, conv=(w == 0.005)),
                          s, w, 'cold', 2026, 3, .1)]
    def warm(s, cfg, donor):
        return synthetic_result(s, 3.0, cfg.omega, conv=True)
    best = scan(weights, ('b2',), lambda w: SimpleNamespace(omega=w), cold, warm,
                lambda donor,cfg: donor.wsr, lambda result,cfg:None, tmp_path)
    assert best['b2', 0.3].source_type == 'forward'
    assert best['b2', 0.3].wsr == 3.0


def test_consider_rejects_nonconverged_and_lower_wsr():
    d = {}
    base = Candidate(synthetic_result('proposed', 1., 2.), 'proposed', .5, 'cold', 2026, 1, .1)
    assert consider(d, base)
    assert not consider(d, Candidate(synthetic_result('proposed', 100., 1., conv=False),
                                    'proposed', .5, 'forward', 2026, 1, .1))
    assert d['proposed', .5] is base


def test_all_four_wrappers_transfer_complete_state(monkeypatch):
    from experiments.run_stage10_bidirectional import run_warm_scheme
    calls = {}
    def fake(name):
        def run(cfg, **kwargs):
            calls[name] = kwargs
            # Mutate passed arrays/resources to make sure source is independent.
            kwargs['initial_resources'].q_matrix[0, 0] = 99
            if 'initial_endpoints' in kwargs:
                kwargs['initial_endpoints'][0, 0] = 99
            if 'initial_tx_positions' in kwargs:
                kwargs['initial_tx_positions'][0, 0] = 99
            return SimpleNamespace(name=name)
        return run
    monkeypatch.setitem(sys.modules, 'algorithm', SimpleNamespace(run_joint_algorithm=fake('proposed')))
    monkeypatch.setitem(sys.modules, 'baselines', SimpleNamespace(
        run_fixed_toma_fp=fake('b2'), run_fpa_upa_fp=fake('b3')))
    monkeypatch.setitem(sys.modules, 'ma_baseline', SimpleNamespace(run_ma_fp=fake('b6')))
    original = SimpleNamespace(
        geometry=SimpleNamespace(endpoints=np.zeros((4, 3)),
                                 tx_positions=np.zeros((4, 3)), rx_positions=np.ones((4, 3))),
        resources=SimpleNamespace(q_matrix=np.ones((2, 2)),
                                  q_ul=np.ones(2), b_ul=np.ones(2), u_s=np.ones((2, 2))),
    )
    exp = SimpleNamespace(ma_region_size=.3, ma_min_spacing=.005,
                          ma_initial_step=.03, ma_initial_jitter_fraction=.5)
    for scheme in SCHEMES:
        donor = Candidate(SimpleNamespace(**vars(original),
            performance=SimpleNamespace(weighted_sum_rate=1.), converged=True),
            scheme, .005, 'cold', 2028, 3, .1)
        result = run_warm_scheme(scheme, SimpleNamespace(), donor, exp)
        assert result.name == scheme
        assert 'initial_resources' in calls[scheme]
        if scheme in ('b2','proposed'):
            assert 'initial_endpoints' in calls[scheme]
        if scheme == 'b6':
            assert 'initial_tx_positions' in calls[scheme]
            assert 'initial_rx_positions' in calls[scheme]
        assert calls[scheme]['initial_resources'].q_matrix[0,0] == 99
    # NONE of the four executions corrupted the donor state.
    assert original.resources.q_matrix[0,0] == 1
    assert original.geometry.endpoints[0,0] == 0
    assert original.geometry.tx_positions[0,0] == 0
