"""Fixed-update training with raw outputs, query validation, and checkpoint predictions."""
import json
import random
from pathlib import Path
import numpy as np
import torch
from src.arr.losses import get_loss
from src.arr.training import (load_trainable_judge, RankingBatchCollator,
    _forward_ranking_batch, _multihead_ranking_loss, set_reproducible_seed)
from src.arr.utils import stable_hash
from .metrics import query_metrics
from .data import read_groups


def predict(model, collator, groups, config, mc_samples=0):
    model.eval()
    if mc_samples:
        if config['epistemic_heads'] != 1:
            raise ValueError('MC dropout is a separate single-model estimator')
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout) or module.__class__.__name__ == 'StableDropout':
                module.train()
    device=next(model.parameters()).device
    rows=[]
    with torch.no_grad():
        for g in groups:
            batch=collator([g]); samples=[]
            for _ in range(max(1,mc_samples)):
                s,_,_=_forward_ranking_batch(model,batch,device,apply_sigmoid=config.get('output_activation','identity')=='sigmoid')
                if s.ndim==2: s=s.unsqueeze(-1)
                samples.append(s[0].T.cpu().numpy())
            scores=np.concatenate(samples,axis=0)
            rows.append({'group_id':g.group_id, 'training_seed':config['seed'],
                'estimator':'mc_dropout' if mc_samples else ('shared_heads' if config['epistemic_heads']>1 else 'single'),
                'candidate_ids':[c.candidate_id for c in g.candidates],
                'data_fingerprint':g.data_fingerprint, 'domain':g.domain,
                'candidate_count':len(g.candidates), 'targets':list(g.scores), 'scores':scores.tolist(),
                'generators':[c.metadata.get('generator') for c in g.candidates],
                'temperature':config.get('temperature',1.),
                **query_metrics(scores,g.scores,config.get('temperature',1.))})
    model.eval()
    return rows


def save_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def train(config, output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    for required in ('train_data','validation_data','test_data','updates','seed','loss'):
        if required not in config: raise ValueError(f'Missing {required}')
    if config.get('output_activation','identity') not in ('identity','sigmoid'):
        raise ValueError('Unknown activation')
    if config['loss'] not in ('mse','listnet','listmle') or config['updates']<1:
        raise ValueError('Invalid objective or update budget')
    if config.get('resume_from'): raise ValueError('Study conditions must start from pretrained weights')
    datasets={s:read_groups(config[s+'_data']) for s in ('train','validation','test')}
    seen=set(); questions=set()
    for s,groups in datasets.items():
        ids={g.group_id for g in groups}; texts={' '.join(g.question.split()) for g in groups}
        if not groups or len(ids)!=len(groups) or seen & ids or questions & texts:
            raise ValueError(f'Empty/duplicate/leaking {s} split')
        seen.update(ids); questions.update(texts)
    config=dict(config, data_hashes={s:stable_hash([g.to_dict() for g in gs]) for s,gs in datasets.items()})
    save_json(output/'config.json',config)
    set_reproducible_seed(config['seed'])
    model,tokenizer,metadata=load_trainable_judge(config)
    metadata['resolved_revision']=getattr(model.config,'_commit_hash',None)
    collator=RankingBatchCollator(tokenizer,config.get('max_length',512))
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['learning_rate'],weight_decay=config.get('weight_decay',.01))
    loss_fn=get_loss(config['loss']); device=next(model.parameters()).device
    tie_samples=int(config.get('listmle_tie_samples',1))
    if tie_samples<1: raise ValueError('Tie sample count must be positive')
    if config['loss']=='listmle' and tie_samples>1:
        base_loss=loss_fn
        loss_fn=lambda s,y,m:torch.stack([base_loss(s,y,m) for _ in range(tie_samples)]).mean()
    rng=random.Random(config['seed']); pool=[]
    batch_size=config.get('group_batch_size',1)
    accum=config.get('gradient_accumulation_steps',8)
    checkpoints=set(config.get('checkpoints',[])) | {config['updates']}
    warmup=max(1,round(config['updates']*config.get('warmup_ratio',.03)))
    best=float('inf'); history=[]; processed=0; candidates_processed=0
    if config.get('save_initial_predictions',False):
        save_json(output/'validation-0.json',predict(model,collator,datasets['validation'],config))
        save_json(output/'test-0.json',predict(model,collator,datasets['test'],config))
    for step in range(1,config['updates']+1):
        model.train(); optimizer.zero_grad(); total=0.
        for _ in range(accum):
            batch_groups=[]
            for _ in range(batch_size):
                if not pool:
                    pool=list(datasets['train']); rng.shuffle(pool)
                batch_groups.append(pool.pop())
            batch=collator(batch_groups)
            scores,targets,mask=_forward_ranking_batch(model,batch,device,apply_sigmoid=config.get('output_activation','identity')=='sigmoid')
            loss=_multihead_ranking_loss(loss_fn,scores,targets,mask)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
            (loss/accum).backward(); total+=loss.item()/accum
            processed+=len(batch_groups); candidates_processed+=sum(len(g.candidates) for g in batch_groups)
        torch.nn.utils.clip_grad_norm_(model.parameters(),config.get('max_grad_norm',1.))
        scale=min(step/warmup, max(0.,(config['updates']-step+1)/max(1,config['updates']-warmup)))
        for param_group in optimizer.param_groups: param_group['lr']=config['learning_rate']*scale
        optimizer.step()
        history.append({'step':step,'loss':total,'groups_processed':processed,'candidates_processed':candidates_processed})
        if step==1 or step%10==0:
            print(json.dumps({'loss':config['loss'],'step':step,'training_loss':total,'groups_processed':processed}),flush=True)
        if step in checkpoints:
            # Evaluation must not change subsequent training's dropout/tie RNG stream.
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                torch.manual_seed(config['seed']+step)
                val=predict(model,collator,datasets['validation'],config)
                test=predict(model,collator,datasets['test'],config)
                score=float(np.mean([r['regret'] for r in val]))
                save_json(output/f'validation-{step}.json',val)
                save_json(output/f'test-{step}.json',test)
                if config.get('mc_samples',0):
                    save_json(output/f'test-mc-{step}.json',predict(model,collator,datasets['test'],config,config['mc_samples']))
            history[-1]['validation_regret']=score
            if score<best:
                best=score
                model.save_pretrained(output/'best',safe_serialization=True)
                tokenizer.save_pretrained(output/'best')
                save_json(output/'selection.json',{'step':step,'validation_regret':score,'criterion':'minimum query regret; earliest exact tie'})
        save_json(output/'history.json',history)
    save_json(output/'manifest.json',{'status':'complete','model':metadata,'config':config,'updates':config['updates'],
        'groups_processed':processed,'candidates_processed':candidates_processed,'best_validation_regret':best})
