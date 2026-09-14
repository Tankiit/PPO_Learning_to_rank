"""Join independent members and apply a selected checkpoint to an untouched domain."""
import json
from pathlib import Path
import numpy as np
from .metrics import query_metrics


def combine_members(paths):
    if len(paths)!=5: raise ValueError('Reference ensemble requires five independent members')
    if len({Path(p).resolve() for p in paths})!=5: raise ValueError('Repeated member files')
    members=[{r['group_id']:r for r in json.loads(Path(p).read_text())} for p in paths]
    if any(m.keys()!=members[0].keys() for m in members): raise ValueError('Mismatched test questions')
    result=[]
    for q in sorted(members[0]):
        rows=[m[q] for m in members]; base=rows[0]
        if len({r['training_seed'] for r in rows})!=5:
            raise ValueError('Reference members require independent training seeds')
        for r in rows:
            if any(r[k]!=base[k] for k in ('candidate_ids','targets','data_fingerprint','temperature')):
                raise ValueError('Mismatched candidate set or temperature')
            if len(r['scores'])!=1: raise ValueError('Each independent member must contribute one score vector')
        scores=np.concatenate([r['scores'] for r in rows],axis=0)
        result.append(dict(base,scores=scores.tolist(),estimator='independent_ensemble',member_seeds=[r['training_seed'] for r in rows],**query_metrics(scores,base['targets'],base['temperature'])))
    return result


def predict_checkpoint(run_dir, groups, mc_samples=0):
    from safetensors.torch import load_file
    from src.arr.training import load_trainable_judge,RankingBatchCollator,set_reproducible_seed
    from .train import predict
    root=Path(run_dir); config=json.loads((root/'config.json').read_text())
    set_reproducible_seed(config['seed'])
    if config.get('qlora'):
        config['resume_from']=str(root/'best')
        model,tokenizer,_=load_trainable_judge(config)
    else:
        model,tokenizer,_=load_trainable_judge(config)
        model.load_state_dict(load_file(str(root/'best'/'model.safetensors')),strict=True)
    return predict(model,RankingBatchCollator(tokenizer,config.get('max_length',512)),groups,config,mc_samples)
