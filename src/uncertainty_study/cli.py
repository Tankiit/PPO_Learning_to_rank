"""python -m src.uncertainty_study.cli --help"""
import argparse
import json
from pathlib import Path
import numpy as np
import yaml
from .data import audit_ds, folds, exposure, nested_dose, read_groups, write_groups
from .metrics import evaluate, query_metrics, probability_metrics, paired_bootstrap, exposure_components
from .train import train, save_json


def prepare_esnli(args):
    # Accept existing constructed groups; never reinterpret tiers as human ratings.
    root=Path(args.output); root.mkdir(parents=True,exist_ok=False)
    seen=set(); manifest={}
    for split in ('train','validation','test'):
        gs=read_groups(getattr(args,split))
        kept=[]
        for g in gs:
            key=' '.join(g.question.split())
            if key not in seen:
                kept.append(g); seen.add(key)
        if not kept: raise ValueError(f'Empty e-SNLI {split}')
        write_groups(root/f'{split}.jsonl',kept)
        manifest[split]={'questions':len(kept),'duplicate_questions_excluded':len(gs)-len(kept)}
        if split=='train':
            for dose in (.1,.25,.5,1.): write_groups(root/f'dose-{dose:g}.jsonl',nested_dose(kept,dose,args.seed))
    save_json(root/'audit.json',{'benchmark':'e-SNLI-derived ranking benchmark','splits':manifest,
        'duplicate_policy':'Keep first normalized question, in train/validation/test precedence'})


def predict_saved(args):
    from .estimators import predict_checkpoint
    save_json(args.output,predict_checkpoint(args.run,read_groups(args.data),args.mc_samples))


def merge_saved(args):
    from .estimators import combine_members
    save_json(args.output,combine_members(args.members))


def plot_saved(args):
    import matplotlib
    matplotlib.use('Agg')
    from .figures import trajectories, exposure_figure
    (trajectories if args.kind=='trajectories' else exposure_figure)(args.inputs,args.output)


def select_hpo(args):
    cfg=yaml.safe_load(Path(args.config).read_text()); bykey={}
    for path in args.runs:
        root=Path(path); c=json.loads((root/'config.json').read_text())
        manifest=json.loads((root/'manifest.json').read_text())
        if c.get('stage')!='hpo' or manifest['status']!='complete':
            raise ValueError('Only completed HPO runs may select learning rates')
        key=f'{c["architecture"]}:{c["fold"]}:{c["loss"]}'
        selection=json.loads((root/'selection.json').read_text())
        bykey.setdefault(key,[]).append((c['learning_rate'],selection['validation_regret'],str(root)))
    chosen={}
    for key,trials in bykey.items():
        rates=[x[0] for x in trials]
        if len(rates)!=len(set(rates)) or set(rates)!=set(cfg['hpo_learning_rates']):
            raise ValueError(f'Incomplete or duplicate equal-budget trials for {key}')
        winner=min(trials,key=lambda x:(x[1],x[0]))
        chosen[key]={'learning_rate':winner[0],'validation_regret':winner[1],'source_run':winner[2]}
    cfg['selected_learning_rates']=chosen
    Path(args.output).write_text(yaml.safe_dump(cfg,sort_keys=False))


def prepare(args):
    out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    groups,audit=audit_ds(args.sources)
    splits=folds(groups,args.seed)
    generators=sorted(audit['question_support'])
    support={g:min(len(exposure(f['train'],g,'full',args.seed)) for f in splits) for g in generators}
    targets=sorted(generators,key=lambda g:(-support[g],g))[:2]
    if any(support[g]<args.min_exposure_questions for g in targets):
        raise ValueError(f'Insufficient matched exposure support: {support}')
    audit.update(exposure_question_support_min_fold=support,targets=targets,
        target_selection='Top two by minimum eligible training-question support across folds; lexical tie-break')
    for i,fold in enumerate(splits):
        root=out/f'fold-{i}'
        for split,gs in fold.items(): write_groups(root/f'{split}.jsonl',gs)
        for dose in (.1,.25,.5,1.):
            write_groups(root/f'dose-{dose:g}.jsonl',nested_dose(fold['train'],dose,args.seed))
        for target in targets:
            for level in ('absent','partial','full'):
                gs=exposure(fold['train'],target,level,args.seed)
                write_groups(root/f'{target}-{level}.jsonl',gs)
                audit.setdefault('exposure_balance',[]).append({'fold':i,'target':target,'level':level,
                    'questions':len(gs),'candidates':sum(len(g.candidates) for g in gs),
                    'target_candidates':sum(c.metadata['generator']==target for g in gs for c in g.candidates),
                    'mean_quality':float(np.mean([c.score for g in gs for c in g.candidates])),
                    'mean_length':float(np.mean([c.metadata['length_words'] for g in gs for c in g.candidates]))})
    save_json(out/'audit.json',audit)
    print(json.dumps(audit,indent=2))


def matrix(args):
    cfg=yaml.safe_load(Path(args.config).read_text()); root=Path(args.data); out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    targets=json.loads((root/'audit.json').read_text())['targets'] if args.dataset=='ds' else []
    fold_ids=range(3) if args.dataset=='ds' else [None]
    specs=[]
    for fold in fold_ids:
        data=root/f'fold-{fold}' if fold is not None else root
        for architecture,model in cfg['models'].items():
            for loss in ('listnet','listmle','mse'):
                for seed in cfg['seeds']:
                    common={**cfg['training'],**model,'loss':loss,'seed':seed,
                            'validation_data':str(data/'validation.jsonl'),'test_data':str(data/'test.jsonl'),
                            'dataset':args.dataset,'fold':fold}
                    specs.append(('core','full',dict(common,train_data=str(data/'train.jsonl'))))
                    if architecture=='deberta':
                        for dose in (.1,.25,.5):
                            specs.append(('dose',str(dose),dict(common,train_data=str(data/f'dose-{dose:g}.jsonl'),dose=dose)))
                    else:
                        specs.append(('replication','dose-0.1',dict(common,train_data=str(data/'dose-0.1.jsonl'),dose=.1)))
                    for target in targets:
                        for level in ('absent','partial','full') if architecture=='deberta' else ('absent','full'):
                            specs.append(('exposure',f'{target}-{level}',dict(common,train_data=str(data/f'{target}-{level}.jsonl'),target=target,exposure=level)))
                    if architecture=='deberta':
                        selected=[('full',str(data/'train.jsonl'))]+[(f'{t}-absent',str(data/f'{t}-absent.jsonl')) for t in targets]
                        for label,path in selected:
                            specs.append(('mc',label,dict(common,train_data=path,epistemic_heads=1,mc_samples=20)))
                            for member in range(5):
                                specs.append(('independent',f'{label}-member-{member}',dict(common,train_data=path,epistemic_heads=1,ensemble_seed=seed,seed=seed*100+member,member=member)))
        # Predefined activation sensitivity and adequately trained dose endpoints.
        for loss in ('listnet','listmle','mse'):
            for seed in cfg['seeds']:
                base={**cfg['training'],**cfg['models']['deberta'],'loss':loss,'seed':seed,
                    'dataset':args.dataset,'fold':fold,'validation_data':str(data/'validation.jsonl'),'test_data':str(data/'test.jsonl')}
                specs.append(('activation','sigmoid',dict(base,train_data=str(data/'train.jsonl'),output_activation='sigmoid')))
                for dose in (.1,1.):
                    steps=base['updates']*cfg.get('endpoint_update_multiplier',3)
                    specs.append(('endpoint',str(dose),dict(base,train_data=str(data/f'dose-{dose:g}.jsonl'),updates=steps,checkpoints=sorted(set(base['checkpoints']+[steps])),dose=dose)))
    if args.stage == 'hpo':
        specs=[('hpo',f'lr-{lr}',dict(c,learning_rate=lr)) for stage,condition,c in specs
               if stage=='core' and c['seed']==cfg['seeds'][0] for lr in cfg['hpo_learning_rates']]
    elif args.stage == 'ties':
        specs=[('ties','eight-permutations',dict(c,listmle_tie_samples=8)) for stage,condition,c in specs
               if stage=='core' and c['architecture']=='encoder' and c['loss']=='listmle']
    elif args.stage == 'pilot':
        specs=[(stage,condition,c) for stage,condition,c in specs if stage=='core' and c['architecture']=='encoder' and c['seed']==cfg['seeds'][0] and c['fold'] in (None,0)]
    elif args.stage != 'all':
        specs=[x for x in specs if x[0]==args.stage]
    index=[]
    for i,(stage,condition,c) in enumerate(specs):
        key=f'{c["architecture"]}:{c["fold"]}:{c["loss"]}'
        if args.stage not in ('pilot','hpo') and 'selected_learning_rates' in cfg:
            if key not in cfg['selected_learning_rates']: raise ValueError(f'Missing selected HPO setting: {key}')
            c['learning_rate']=cfg['selected_learning_rates'][key]['learning_rate']
        identifier=f'{i:04d}-{stage}-{c["architecture"]}-{c["loss"]}-f{c["fold"]}-s{c["seed"]}-{condition}'
        c.update(stage=stage,condition=condition)
        path=out/(identifier+'.json'); save_json(path,c)
        index.append({'id':identifier,'stage':stage,'config':str(path)})
    save_json(out/'index.json',index)
    print(f'Wrote {len(index)} run configurations; no training launched')


def diagnostics(args):
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.special import softmax
    from src.arr.losses import listmle_loss,listnet_loss,masked_mse
    out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(args.seed)
    s=rng.normal(size=(5,8)); y=np.linspace(0,1,8); shift=rng.normal(size=(5,1))
    rows=[]
    for scale in (0,1,5,20,100):
        shifted=s+scale*shift
        row={'offset_scale':scale,**query_metrics(shifted,y)}
        row['ranking_unchanged']=bool(np.array_equal(np.argsort(s,axis=1),np.argsort(shifted,axis=1)))
        for name,fn in [('listnet',listnet_loss),('listmle',listmle_loss),('mse',masked_mse)]:
            torch.manual_seed(args.seed)
            row[name]=float(fn(torch.tensor(shifted),torch.tensor(np.tile(y,(5,1))),torch.ones((5,8),dtype=torch.bool)))
        rows.append(row)
    p=softmax(s,axis=1); mean=p.mean(0); contraction=[]
    for a in np.geomspace(1e-5,1,80): contraction.append({'alpha':float(a),**probability_metrics(mean+a*(p-mean))})
    save_json(out/'invariance.json',rows); save_json(out/'contraction.json',contraction)
    fig,axes=plt.subplots(1,3,figsize=(13,3.5),layout='constrained')
    for key in ('raw_variance','centered_variance','js'):
        axes[0].plot([r['offset_scale'] for r in rows],[r[key] for r in rows],label=key)
    axes[0].set(yscale='log',xlabel='Member-specific offset scale',ylabel='Disagreement'); axes[0].legend()
    axes[0].set_title('All member rankings unchanged')
    for key in ('listnet','listmle','mse'):
        axes[1].plot([r['offset_scale'] for r in rows],[r[key]/rows[0][key] for r in rows],label=key)
    axes[1].set(yscale='log',xlabel='Member-specific offset scale',ylabel='Loss / original loss'); axes[1].legend()
    axes[2].plot([r['covariance_trace'] for r in contraction],[r['participation_ratio'] for r in contraction])
    axes[2].set(xscale='log',ylim=(0,4),xlabel='Probability covariance trace',ylabel='Participation ratio',title='Controlled contraction (synthetic)')
    fig.savefig(out/'diagnostics.png',dpi=180); fig.savefig(out/'diagnostics.pdf'); plt.close(fig)
    print(f'Saved mathematical diagnostics to {out}')


def analysis(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=json.loads(Path(args.predictions).read_text())
    # Always recompute metrics from member outputs.
    rows=[dict(r,**query_metrics(r['scores'],r['targets'],r.get('temperature',1.))) for r in rows]
    out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    result=evaluate(rows,permutations=args.permutations)
    save_json(out/'query_metrics.json',rows); save_json(out/'risk.json',result)
    fig,ax=plt.subplots(figsize=(6,4),layout='constrained')
    for key in ('js','entropy','negative_margin','raw_variance','centered_variance','random','oracle'):
        ax.plot(result[key]['coverage'],result[key]['risk'],label=key)
    ax.set(xlabel='Coverage',ylabel='Top-1 regret'); ax.legend(fontsize=8)
    fig.savefig(out/'risk-coverage.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4),layout='constrained')
    valid=[r for r in rows if r['participation_ratio'] is not None and r['covariance_trace']>0]
    ax.scatter([r['covariance_trace'] for r in valid],[r['participation_ratio'] for r in valid],s=10,alpha=.5)
    ax.set(xscale='log',xlabel='Probability covariance trace',ylabel='Participation ratio')
    fig.savefig(out/'shape-magnitude.png',dpi=180); plt.close(fig)


def compare(args):
    before={r['group_id']:r for r in json.loads(Path(args.before).read_text())}
    after={r['group_id']:r for r in json.loads(Path(args.after).read_text())}
    if before.keys()!=after.keys(): raise ValueError('Test questions changed')
    a=[]; b=[]
    for q in sorted(before):
        x,y=before[q],after[q]
        if any(x[k]!=y[k] for k in ('candidate_ids','targets','generators','data_fingerprint')):
            raise ValueError(f'Test candidates/reference changed: {q}')
        if args.target:
            cx,cy=exposure_components(x,args.target),exposure_components(y,args.target)
            if cx is None: continue
        else:
            cx=query_metrics(x['scores'],x['targets'],x.get('temperature',1.)); cy=query_metrics(y['scores'],y['targets'],y.get('temperature',1.))
        a.append(cx); b.append(cy)
    keys=('target','other','pair_error','regret','js') if args.target else ('js','regret','covariance_trace')
    result={}
    for key in keys:
        valid=[(x[key],y[key]) for x,y in zip(a,b) if x[key] is not None and y[key] is not None]
        if valid:
            left,right=zip(*valid)
            result[key]={'before_mean':float(np.mean(left)),'after_mean':float(np.mean(right)),**paired_bootstrap(left,right)}
    if args.target:
        result['target_specific_reduction']=paired_bootstrap([x['target']-x['other'] for x in a],[y['target']-y['other'] for y in b])
    save_json(args.output,result)


def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare-esnli'); p.add_argument('--train',required=True); p.add_argument('--validation',required=True); p.add_argument('--test',required=True); p.add_argument('--output',required=True); p.add_argument('--seed',type=int,default=2026); p.set_defaults(func=prepare_esnli)
    p=sub.add_parser('predict'); p.add_argument('--run',required=True); p.add_argument('--data',required=True); p.add_argument('--output',required=True); p.add_argument('--mc-samples',type=int,default=0); p.set_defaults(func=predict_saved)
    p=sub.add_parser('combine'); p.add_argument('members',nargs=5); p.add_argument('--output',required=True); p.set_defaults(func=merge_saved)
    p=sub.add_parser('plot'); p.add_argument('inputs',nargs='+'); p.add_argument('--kind',choices=['trajectories','exposure'],required=True); p.add_argument('--output',required=True); p.set_defaults(func=plot_saved)
    p=sub.add_parser('select-hpo'); p.add_argument('runs',nargs='+'); p.add_argument('--config',default='configs/uncertainty_study/study.yaml'); p.add_argument('--output',required=True); p.set_defaults(func=select_hpo)
    p=sub.add_parser('prepare-ds'); p.add_argument('sources',nargs='+'); p.add_argument('--output',required=True); p.add_argument('--seed',type=int,default=2026); p.add_argument('--min-exposure-questions',type=int,default=30); p.set_defaults(func=prepare)
    p=sub.add_parser('matrix'); p.add_argument('--config',default='configs/uncertainty_study/study.yaml'); p.add_argument('--data',required=True); p.add_argument('--dataset',choices=['ds','esnli'],required=True); p.add_argument('--output',required=True); p.add_argument('--stage',default='pilot',choices=['pilot','hpo','ties','core','dose','exposure','replication','mc','independent','activation','endpoint','all']); p.set_defaults(func=matrix)
    p=sub.add_parser('diagnostics'); p.add_argument('--output',required=True); p.add_argument('--seed',type=int,default=2026); p.set_defaults(func=diagnostics)
    p=sub.add_parser('train'); p.add_argument('--config',required=True); p.add_argument('--output',required=True); p.set_defaults(func=lambda a:train(json.loads(Path(a.config).read_text()),a.output))
    p=sub.add_parser('analyze'); p.add_argument('--predictions',required=True); p.add_argument('--output',required=True); p.add_argument('--permutations',type=int,default=200); p.set_defaults(func=analysis)
    p=sub.add_parser('compare'); p.add_argument('--before',required=True); p.add_argument('--after',required=True); p.add_argument('--target'); p.add_argument('--output',required=True); p.set_defaults(func=compare)
    args=parser.parse_args(); args.func(args)

if __name__=='__main__': main()
