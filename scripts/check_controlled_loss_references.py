"""Execute the inspected upstream loss methods without importing its training stack.

Usage: python -m scripts.check_controlled_loss_references --ptranking /path/to/clone
This loads only the named AST function definitions from an explicitly supplied
PT-Ranking checkout. No copied implementation is vendored in this repository.
"""
import argparse
import ast
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from src.arr.losses import get_loss


def load_function(path,name,namespace,class_name=None):
    tree=ast.parse(Path(path).read_text())
    nodes=tree.body
    if class_name:
        nodes=next(n for n in nodes if isinstance(n,ast.ClassDef) and n.name==class_name).body
    fn=next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--ptranking',required=True)
    parser.add_argument('--output',default='runs/uncertainty_study/loss-reference-check.json'); args=parser.parse_args()
    root=Path(args.ptranking); code=root/'ptranking/ltr_adhoc'
    ns={'torch':torch,'F':F}
    load_function(code/'util/sampling_utils.py','arg_shuffle_ties',ns)
    mse=load_function(code/'pointwise/rank_mse.py','rankMSE_loss_function',ns)
    upstream={name:load_function(code/f'listwise/{name}.py','custom_loss_function',dict(ns),class_name=cls)
        for name,cls in [('listnet','ListNet'),('listmle','ListMLE')]}
    no_op=SimpleNamespace(optimizer=SimpleNamespace(zero_grad=lambda:None,step=lambda:None),device='cpu')
    rng=np.random.default_rng(2026); results=[]
    for batch,n in [(1,3),(3,7),(2,12)]:
        for tied in (False,True):
            targets=np.tile(np.arange(n)/max(1,n-1),(batch,1))
            if tied: targets=np.round(targets*2)/2
            y=torch.tensor(targets,dtype=torch.float64)
            for loss in ('listnet','listmle','mse'):
                s=torch.tensor(rng.normal(size=(batch,n)),dtype=torch.float64,requires_grad=True)
                reference=s.detach().clone().requires_grad_(True)
                torch.manual_seed(7)
                ours=get_loss(loss)(s,y,torch.ones_like(y,dtype=torch.bool)); ours.backward()
                torch.manual_seed(7)
                if loss=='mse':
                    theirs=mse(reference,y)/n; theirs.backward()
                else:
                    raw=upstream[loss](no_op,reference,y)
                    theirs=raw/batch
                    reference.grad/=batch
                values=float(abs(ours.item()-theirs.item()))
                gradients=float((s.grad-reference.grad).abs().max())
                results.append({'loss':loss,'batch':batch,'candidates':n,'exact_target_ties':tied,
                    'loss_absolute_difference':values,'max_gradient_difference':gradients})
                if values>1e-10 or gradients>1e-10:
                    raise AssertionError(results[-1])
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'upstream_commit':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        'normalization':{'listnet':'PT batch sum / batch size','listmle':'PT batch sum / batch size','mse':'PT per-query candidate sum / candidate count'},
        'comparisons':results},indent=2)+'\n')
    print(f'{len(results)} forward-and-gradient comparisons passed; {out}')

if __name__=='__main__': main()
