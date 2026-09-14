import itertools
import json
from dataclasses import replace
import numpy as np
import pytest
import torch
from scipy.special import softmax
from src.arr.losses import listmle_loss, listnet_loss, masked_mse
from src.arr.schema import Candidate, RankingGroup
from src.uncertainty_study.data import folds, exposure, nested_dose, audit_ds
from src.uncertainty_study.metrics import query_metrics, probability_metrics, risk_coverage, tie_ndcg, exposure_components


def groups(n=24):
    return [RankingGroup(str(i),'train','d',f'q {i}',tuple(Candidate(f'{i}-{p}-{g}',f'text {i} {p} {g}',g/4,'human',{'generator':str(g),'prompt':str(p)}) for p in range(3) for g in range(4)),'fingerprint') for i in range(n)]


def test_invariance_and_gradients():
    s=torch.tensor([[1.,-2.,.4,99.],[.3,2.,-1.,99.]],dtype=torch.float64,requires_grad=True)
    y=torch.tensor([[1.,.5,.5,0.],[0.,1.,.5,0.]])
    m=torch.tensor([[1,1,1,0],[1,1,1,0]],dtype=torch.bool)
    shift=torch.tensor([[12.],[-34.]])
    for fn in (listnet_loss,listmle_loss):
        torch.manual_seed(5); before=fn(s,y,m)
        torch.manual_seed(5); after=fn(s+shift,y,m)
        assert torch.allclose(before,after,atol=1e-10)
        grad=torch.autograd.grad(before,s,retain_graph=True)[0]
        assert torch.allclose(grad.sum(1),torch.zeros(2,dtype=s.dtype),atol=1e-10)
        assert (grad[:,-1]==0).all()
    assert not torch.allclose(masked_mse(s,y,m),masked_mse(s+shift,y,m))
    x=s.detach().numpy()[:,:3]
    a,b=query_metrics(x,[1,.5,0]),query_metrics(x+shift.numpy(),[1,.5,0])
    for key in ('js','covariance_trace','centered_variance','participation_ratio','regret'):
        assert a[key]==pytest.approx(b[key],abs=1e-10)
    assert a['raw_variance']!=pytest.approx(b['raw_variance'])


def test_listmle_ties_average_and_permutation():
    s=torch.tensor([[2.,0.,-1.]],dtype=torch.float64); y=torch.ones_like(s); mask=y.bool()
    exact=[]
    for perm in itertools.permutations(range(3)):
        ordered=s[0,list(perm)]
        exact.append((torch.logcumsumexp(ordered.flip(0),0).flip(0)-ordered).sum().item())
    torch.manual_seed(42)
    observed=[listmle_loss(s,y,mask).item() for _ in range(3000)]
    assert np.mean(observed)==pytest.approx(np.mean(exact),abs=.08)
    assert len(set(round(x,5) for x in observed))==6
    assert listmle_loss(s[:,:1],y[:,:1],mask[:,:1]).item()==0


def test_contraction():
    p=softmax(np.random.default_rng(0).normal(size=(5,8)),axis=1)
    a=probability_metrics(p); b=probability_metrics(p.mean(0)+.01*(p-p.mean(0)))
    assert b['covariance_trace']==pytest.approx(a['covariance_trace']*.0001)
    assert b['participation_ratio']==pytest.approx(a['participation_ratio'])
    assert probability_metrics(np.tile(p.mean(0),(5,1)))['participation_ratio'] is None


def test_risk_and_ranking_ties():
    y=np.array([0.,.4,1.]); pred=np.ones(3)
    expected=np.mean([tie_ndcg(y,np.array(perm),k=2) for perm in itertools.permutations(range(3))])
    assert tie_ndcg(y,pred,k=2)==pytest.approx(expected)
    assert risk_coverage([0,1,0,1],[0,0,0,0])['risk']==[.5]*4
    assert risk_coverage([0,1,0,1],[0,1,0,1])['aurc']<.5


def test_paired_data():
    gs=groups(); fs=folds(gs)
    assert set.union(*[{g.group_id for g in f['test']} for f in fs])=={g.group_id for g in gs}
    for f in fs:
        sets=[{g.group_id for g in f[k]} for k in ('train','validation','test')]
        assert not any(a & b for a,b in itertools.combinations(sets,2))
    previous=set()
    for dose in (.1,.25,.5,1):
        current={g.group_id for g in nested_dose(gs,dose,42)}
        assert previous<=current; previous=current
    conditions=[exposure(gs,'0',level) for level in ('absent','partial','full')]
    assert all([g.group_id for g in c]==[g.group_id for g in gs] for c in conditions)
    assert all(len(g.candidates)==9 for c in conditions for g in c)
    counts=[sum(x.metadata['generator']=='0' for g in c for x in g.candidates) for c in conditions]
    assert counts==[0,36,72]
    for i in range(len(gs)):
        assert all(sorted(x.metadata['prompt'] for x in c[i].candidates)==['0']*3+['1']*3+['2']*3 for c in conditions)


def test_audit_merges_duplicates_and_keeps_provenance(tmp_path):
    row={'id':'a','qid':'q','question':'Question','dataset':'D','student_model':'g','student_prompt':'p','student_explanation':'explanation','explanation_annotations':[{'worker':'w','explanation_score':3}]}
    other=dict(row,id='b',student_model='h')
    paths=[tmp_path/'a.jsonl',tmp_path/'b.jsonl']
    paths[0].write_text(json.dumps(row)+'\n'+json.dumps(other)+'\n')
    paths[1].write_text(json.dumps(row)+'\n')
    gs,audit=audit_ds(paths)
    assert len(gs)==1 and len(gs[0].candidates)==2
    assert audit['counts']['duplicate_generation_rows']==1
    assert max(len(c.metadata['sources']) for c in gs[0].candidates)==2


def test_exposure_components():
    record={'scores':[[1,0,1],[0,0,1]],'targets':[1,0,.5],'generators':['g','h','h']}
    c=exposure_components(record,'g')
    assert c['target']>=0 and c['other']>=0 and 0<=c['pair_error']<=1


def test_encoder_training_end_to_end(tmp_path):
    from transformers import DebertaV2Config, DebertaV2ForSequenceClassification, BertTokenizer
    from src.uncertainty_study.data import write_groups
    from src.uncertainty_study.train import train
    base=tmp_path/'tiny-deberta'; base.mkdir()
    vocab=base/'vocab.txt'
    vocab.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nq\ntext\n')
    tokenizer=BertTokenizer(vocab=str(vocab)); tokenizer.save_pretrained(base)
    model=DebertaV2ForSequenceClassification(DebertaV2Config(vocab_size=7,hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=32,max_position_embeddings=64,num_labels=5))
    model.save_pretrained(base)
    gs=groups(6)
    paths={}
    for split,selected in [('train',gs[:2]),('validation',gs[2:4]),('test',gs[4:])]:
        path=tmp_path/(split+'.jsonl'); write_groups(path,selected); paths[split+'_data']=str(path)
    for loss in ('listnet','listmle','mse'):
        output=tmp_path/loss
        train({**paths,'base_model':str(base),'architecture':'encoder','qlora':False,'dtype':'float32','device':'cpu',
            'local_files_only':True,'epistemic_heads':5,'epistemic_hidden_dim':8,'updates':2,'checkpoints':[1,2],
            'seed':42,'loss':loss,'learning_rate':.001,'gradient_accumulation_steps':1,'max_length':32},output)
        records=json.loads((output/'test-2.json').read_text())
        assert np.asarray(records[0]['scores']).shape==(5,12)
        assert (output/'best'/'model.safetensors').exists()
        assert json.loads((output/'manifest.json').read_text())['status']=='complete'


def test_decoder_single_mlp_mc_and_checkpoint_restore(tmp_path):
    from transformers import GPT2Config,GPT2ForSequenceClassification,BertTokenizer
    from src.uncertainty_study.data import write_groups
    from src.uncertainty_study.train import train
    from src.uncertainty_study.estimators import predict_checkpoint
    base=tmp_path/'tiny-decoder'; base.mkdir()
    vocab=base/'vocab.txt'; vocab.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nq\ntext\n')
    BertTokenizer(vocab=str(vocab)).save_pretrained(base)
    GPT2ForSequenceClassification(GPT2Config(vocab_size=7,n_embd=16,n_layer=1,n_head=2,n_positions=64,num_labels=1,pad_token_id=0)).save_pretrained(base)
    gs=groups(6); paths={}
    for split,selected in [('train',gs[:2]),('validation',gs[2:4]),('test',gs[4:])]:
        path=tmp_path/(split+'.jsonl'); write_groups(path,selected); paths[split+'_data']=str(path)
    out=tmp_path/'trained'
    train({**paths,'base_model':str(base),'architecture':'decoder','qlora':False,'dtype':'float32','device':'cpu',
        'local_files_only':True,'epistemic_heads':1,'scalar_mlp_head':True,'epistemic_hidden_dim':8,
        'updates':1,'seed':42,'loss':'listmle','learning_rate':.001,'gradient_accumulation_steps':1,'max_length':32,'mc_samples':3},out)
    saved=json.loads((out/'test-1.json').read_text())
    restored=predict_checkpoint(out,gs[4:])
    assert np.allclose(saved[0]['scores'],restored[0]['scores'])
    mc=json.loads((out/'test-mc-1.json').read_text())
    assert np.asarray(mc[0]['scores']).shape==(3,12)
    assert np.ptp(np.asarray(mc[0]['scores']),axis=0).max()>0
