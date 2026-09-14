"""Query-level R1–R4 measurements. All covariance uses population normalization."""
import numpy as np
from scipy.special import softmax, xlogy
from scipy.stats import spearmanr


def probability_metrics(p):
    p = np.asarray(p, dtype=float)
    if p.ndim != 2 or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1):
        raise ValueError('Expected finite member-by-candidate probability vectors')
    mean = p.mean(0)
    residual = p - mean
    cov = residual.T @ residual / len(p)
    trace = float(np.trace(cov))
    squared = float(np.square(cov).sum())
    return {'js': float(np.mean(np.sum(xlogy(p, p) - xlogy(p, mean), axis=1))),
            'probability_width': float(np.ptp(p, axis=0).mean()),
            'covariance_trace': trace,
            'participation_ratio': trace**2 / squared if trace > 1e-24 else None,
            'entropy': float(-xlogy(mean, mean).sum()),
            'negative_margin': float(-np.diff(np.sort(mean)[-2:])[0]) if len(mean)>1 else 0.,
            'candidate_variance': np.diag(cov).tolist(),
            'candidate_width': np.ptp(p, axis=0).tolist()}


def tie_ndcg(y, prediction, k=5):
    """Expected DCG under uniform ordering of exact prediction ties, including cutoff."""
    y, prediction = np.asarray(y), np.asarray(prediction)
    n = len(y)
    discount = np.zeros(n)
    discount[:min(k,n)] = 1 / np.log2(np.arange(2, min(k,n)+2))
    gain = np.exp2(y) - 1
    order = np.argsort(-prediction, kind='stable')
    sorted_p = prediction[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_p)) + 1, n]
    dcg = sum(gain[order[a:b]].mean() * discount[a:b].sum() for a,b in zip(starts[:-1], starts[1:]))
    ideal = np.sort(gain)[::-1] @ discount
    return float(dcg / ideal) if ideal > 0 else 1.


def regret(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    # Expected regret for uniform selection among exact maxima.
    return float(y.max() - y[prediction == prediction.max()].mean())


def query_metrics(scores, targets, temperature=1.):
    s, y = np.asarray(scores, dtype=float), np.asarray(targets, dtype=float)
    if s.ndim != 2 or s.shape[1] != len(y) or s.shape[0] < 1 or len(y)<2 or not np.isfinite(s).all() or not np.isfinite(y).all() or (y<0).any() or (y>1).any() or not np.isfinite(temperature) or temperature<=0:
        raise ValueError('Invalid scores, targets, or temperature')
    p = softmax(s / temperature, axis=1)
    mean = p.mean(0)
    centered = s - s.mean(1, keepdims=True)
    return {**probability_metrics(p), 'raw_variance': float(s.var(0).mean()),
            'raw_width': float(np.ptp(s, axis=0).mean()),
            'centered_variance': float(centered.var(0).mean()),
            'regret': regret(y, mean), 'ndcg_error': 1-tie_ndcg(y, mean),
            'mean_score_regret': regret(y, s.mean(0)),
            'member_regret': [regret(y, row) for row in p],
            'member_ndcg': [tie_ndcg(y,row) for row in p],
            'single_entropy': float(-xlogy(p[0],p[0]).sum()),
            'single_negative_margin': float(-np.diff(np.sort(p[0])[-2:])[0])}


def risk_coverage(errors, uncertainty):
    """Discrete AURC at coverages 1/n,...,1; exact uncertainty ties averaged."""
    e, u = np.asarray(errors, float), np.asarray(uncertainty, float)
    if len(e)==0 or e.shape != u.shape or not np.isfinite(e).all() or not np.isfinite(u).all():
        raise ValueError('Invalid risk arrays')
    order = np.argsort(u, kind='stable'); ordered = e[order].copy(); su = u[order]
    edges = np.r_[0, np.flatnonzero(np.diff(su))+1, len(e)]
    for a,b in zip(edges[:-1],edges[1:]):
        ordered[a:b] = ordered[a:b].mean()
    risk = np.cumsum(ordered) / np.arange(1,len(e)+1)
    return {'coverage': (np.arange(1,len(e)+1)/len(e)).tolist(), 'risk': risk.tolist(), 'aurc': float(risk.mean())}


BASELINES = ('js','entropy','negative_margin','raw_variance','centered_variance','single_entropy','single_negative_margin')


def evaluate(rows, seed=2026, permutations=200):
    rng = np.random.default_rng(seed)
    e = np.array([r['regret'] for r in rows])
    out = {}
    for key in BASELINES:
        u = np.array([r[key] for r in rows])
        out[key] = risk_coverage(e,u)
        out[key]['spearman'] = float(spearmanr(e,u).statistic) if np.ptp(e)>0 and np.ptp(u)>0 else None
    out['oracle'] = risk_coverage(e,e)
    out['random'] = risk_coverage(e,np.zeros(len(e)))
    u = np.array([r['js'] for r in rows])
    strata = {}
    for i,r in enumerate(rows):
        strata.setdefault((r['domain'],r['candidate_count']),[]).append(i)
    nulls = {'global': [], 'domain_candidate_count': []}
    for _ in range(permutations):
        nulls['global'].append(risk_coverage(e,rng.permutation(u))['aurc'])
        shuffled = u.copy()
        for idx in strata.values():
            shuffled[idx] = rng.permutation(u[idx])
        nulls['domain_candidate_count'].append(risk_coverage(e,shuffled)['aurc'])
    out['permutation_control'] = {key: {'aurc': vals, 'improvement_over_js': float(np.mean(vals)-out['js']['aurc']),
        'one_sided_p': float((1+sum(v<=out['js']['aurc'] for v in vals))/(1+len(vals)))} for key,vals in nulls.items()}
    js_pairs=np.column_stack([e,u])
    out['paired_aurc_improvement']={key:paired_bootstrap(
        np.column_stack([e,[r[key] for r in rows]]),js_pairs,
        statistic=lambda x:risk_coverage(x[:,0],x[:,1])['aurc'],seed=seed)
        for key in BASELINES if key!='js'}
    return out


def paired_bootstrap(left, right, statistic=np.mean, samples=1000, seed=2026):
    """Paired question resampling; caller must align questions first."""
    a,b = np.asarray(left), np.asarray(right)
    if a.shape != b.shape or not len(a):
        raise ValueError('Paired nonempty arrays required')
    rng=np.random.default_rng(seed)
    values=[]
    for _ in range(samples):
        ix=rng.integers(0,len(a),len(a))
        values.append(float(statistic(a[ix])-statistic(b[ix])))
    return {'difference': float(statistic(a)-statistic(b)), 'ci95': np.quantile(values,[.025,.975]).tolist(), 'questions':len(a)}


def exposure_components(record, target):
    s=np.asarray(record['scores']); y=np.asarray(record['targets'])
    mask=np.array([g==target for g in record['generators']])
    if not mask.any() or mask.all():
        return None
    p=softmax(s/record.get('temperature',1.),axis=1)
    v=p.var(0); mean=p.mean(0)
    i,j=np.triu_indices(len(y),1)
    relevant=(mask[i]|mask[j]) & (y[i]!=y[j])
    delta=(mean[i]-mean[j]) * (y[i]-y[j])
    pair_error=np.where(delta==0,.5,delta<0)
    return {'target':float(v[mask].mean()), 'other':float(v[~mask].mean()),
        'pair_error':float(pair_error[relevant].mean()) if relevant.any() else None,
        'regret':regret(y,mean), 'js':probability_metrics(p)['js']}
