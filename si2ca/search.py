"""Compare a completed strategy evaluation with its incumbent; no autonomous API calls."""
import argparse
import json
from si2ca.results import load_results, summary


def decide(old, new, delta=7, epsilon=2, ratio=.9, context_audit=False):
    if set(old)!=set(new) or len(new)!=192:
        raise ValueError('Promotion requires the same complete 192-task split')
    if not summary(old,192)['final'] or not summary(new,192)['final']:
        raise ValueError('Resolve infrastructure failures before promotion')
    if {r.get('harness_protocol') for r in old.values()} != {r.get('harness_protocol') for r in new.values()}:
        raise ValueError('Promotion requires matching harness versions; rerun both arms with the same protocol')
    change=sum(r.get('reward',0)>=.999 for r in new.values())-sum(r.get('reward',0)>=.999 for r in old.values())
    oldturn=sum(r['turns'] for r in old.values())
    turnratio=sum(r['turns'] for r in new.values())/oldturn
    # gen_completion_tokens includes discarded candidates: it is NOT retained-trajectory cost.
    tokenratio=None
    if all('traj_tokens' in r for r in [*old.values(),*new.values()]):
        denom=sum(r['traj_tokens'] for r in old.values())
        tokenratio=sum(r['traj_tokens'] for r in new.values())/denom if denom else None
    efficient=min([turnratio]+([tokenratio] if tokenratio is not None else []))<=ratio
    accept=context_audit and (change>=delta or (abs(change)<delta and change>=-epsilon and efficient))
    return {'paired_solved_change':change,'turn_ratio':turnratio,'retained_token_ratio':tokenratio,
            'context_audit_passed':context_audit,'accept':accept}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--incumbent',required=True)
    p.add_argument('--candidate',required=True)
    p.add_argument('--delta',type=int,default=7)
    p.add_argument('--epsilon',type=int,default=2)
    p.add_argument('--ratio',type=float,default=.9)
    p.add_argument('--context-audit-passed',action='store_true',help='Reviewer has verified no PI injection or task-identity rules')
    a=p.parse_args()
    print(json.dumps(decide(load_results(a.incumbent),load_results(a.candidate),a.delta,a.epsilon,a.ratio,a.context_audit_passed),indent=2))
