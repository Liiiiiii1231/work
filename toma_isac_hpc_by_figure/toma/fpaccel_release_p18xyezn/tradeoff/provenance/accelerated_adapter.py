"""Isolated adapters around unchanged shared optimization modules."""
from contextlib import contextmanager,ExitStack
from unittest.mock import patch
from dataclasses import asdict
import numpy as np
import algorithms as shared
from fp_acceleration import DEFAULT_POLICY,solve_accelerated


@contextmanager
def main_resource_solver(policy):
    calls=[];position_records=[];last_rcg=[]
    original_positions=shared.update_positions_once_fp_aware

    def solve(channels,resources,cfg,verbose=False):
        rr,history,diagnostics=solve_accelerated(channels,resources,cfg,policy=policy,verbose=verbose)
        diagnostics['outer_iteration']=len(calls)+1
        diagnostics['after_position_iteration']=len(calls) if calls else None
        calls.append(diagnostics)
        return rr,history

    def positions(*args,**kwargs):
        records=[];kwargs['diagnostic_records']=records
        result=original_positions(*args,**kwargs)
        position_records.append(records);last_rcg[:]=[result[4]]
        return result

    with ExitStack() as stack:
        stack.enter_context(patch.object(shared,'solve_inner_problem',solve))
        stack.enter_context(patch.object(shared,'update_positions_once_fp_aware',positions))
        stack.enter_context(patch.object(shared,'_inner_history_converged',
            lambda history,cfg:bool(calls and calls[-1]['converged'])))
        yield calls,position_records,last_rcg


def attach_diagnostics(result,scheme,calls,position_records,last_rcg):
    if scheme in ('b2','b3'):
        result.iterations=sum(r['actual_fp_iterations'] for r in calls)
        result.converged=bool(calls[-1]['converged'])
    if scheme=='proposed':
        result.fp_aware_diagnostics=[[asdict(r) for r in sweep] for sweep in position_records]
    result.accelerated_inner_diagnostics=calls
    result.resource_solver_policy=DEFAULT_POLICY.copy()
    if last_rcg:
        result.final_rcg_state=last_rcg[0]


def rcg_arrays(state):
    data={}
    for name in ('previous_gradients','previous_directions'):
        values=getattr(state,name)
        data[name]=np.asarray([np.zeros(3) if x is None else x for x in values])
        data[name+'_valid']=np.asarray([x is not None for x in values])
    return data
