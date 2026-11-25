"""P2: empirical unlearning tests; NO certification of forgetting."""
import csv, json, math, time
from pathlib import Path
import numpy as np
import torch
from avse_unlearning import make_model, infer_architecture, SR, mix_babble

def auc_binary(pos, neg):
    # Greater attack scores imply class 'member'. Ties receive half credit.
    if not len(pos) or not len(neg):return float('nan')
    p=np.asarray(pos)[:,None];n=np.asarray(neg)[None,:]
    return float(((p>n).sum()+0.5*(p==n).sum())/(p.size*n.size))

def losses(model_path, rows, pool, vault, seed=42, device='cpu', max_items=128):
    model=make_model(infer_architecture(model_path)).to(device)
    model.load_state_dict(torch.load(model_path,map_location='cpu',weights_only=True))
    model.eval(); scores=[]
    with torch.no_grad():
        for row in rows[:max_items]:
            audio, frames=vault.load(row['id']);audio=np.asarray(audio,dtype='float32')
            noisy=mix_babble(audio,pool,row['id'],seed,0.0,vault)
            x=torch.tensor(noisy,device=device).unsqueeze(0)
            v=torch.tensor(frames.astype('float32')/255.0,device=device).unsqueeze(0).unsqueeze(2)
            y=model(x,v)
            l=float(torch.nn.functional.l1_loss(y,torch.tensor(audio,device=device).unsqueeze(0)))
            scores.append((row['id'],l))
    return scores

def run_audit(root, models, retained, heldout, vault, max_items=128):
    root=Path(root);results=[]
    pool=retained
    for name,path in models.items():
        if not Path(path).exists():continue
        start=time.monotonic()
        a=losses(path,retained,pool,vault,max_items=max_items)
        b=losses(path,heldout,pool,vault,max_items=max_items)
        # Loss-based membership *proxy* on retained vs test samples, not withdrawn subjects.
        auc=auc_binary([-v for _,v in a],[-v for _,v in b])
        results.append({'model':name,'retained_mean_l1':np.mean([v for _,v in a]),
                        'heldout_mean_l1':np.mean([v for _,v in b]),
                        'membership_proxy_auc':auc,'audit_seconds':time.monotonic()-start,
                        'n_retained':len(a),'n_test':len(b)})
    if not results:return []
    with (root/'unlearning_privacy_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    (root/'unlearning_audit_scope.json').write_text(json.dumps({
        'measurement':'Loss-threshold membership proxy distinguishing retained training from untouched held-out samples.',
        'not_measured':'Deleted-sample membership post-revocation; certified removal; independence of model operator.',
        'limits':'Not a calibrated state-of-the-art MIA, and cannot establish that withdrawn samples are forgotten.'},indent=2))
    return results
