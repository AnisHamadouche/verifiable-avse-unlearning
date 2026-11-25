#!/usr/bin/env python3
"""Read-only reproducible IEEE-style report generator for AVSE Privacy Lab.

Supports Streamlit metrics_<model>.csv and CLI per_clip_metrics.csv.
Never reads media, plaintext, encryption keys or model checkpoints.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

LABELS = {
    'si_snri': ('SI-SNR improvement', 'dB', True),
    'output_si_snr': ('Output SI-SNR', 'dB', True),
    'output_stoi': ('STOI', '', True),
    'output_estoi': ('ESTOI', '', True),
    'stoi_improvement': ('STOI improvement', '', True),
    'estoi_improvement': ('ESTOI improvement', '', True),
    'output_pesq_wb': ('PESQ-WB', '', True),
    'pesq_wb_improvement': ('PESQ-WB improvement', '', True),
    'snri': ('SNR improvement', 'dB', True),
    'output_snr': ('Output SNR', 'dB', True),
    'output_segmental_snr': ('Segmental SNR', 'dB', True),
    'output_rmse': ('RMSE', '', False),
    'output_mae': ('MAE', '', False),
}
ORDER = ['original', 'retrained', 'finetuned']
MARKERS = ['o', 's', '^', 'D', 'x', '+']
STYLES = ['-', '--', '-.', ':']

def config():
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['DejaVu Serif', 'Times New Roman'],
        'font.size': 7.5, 'axes.labelsize': 8, 'axes.titlesize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
        'lines.linewidth': 1.2, 'lines.markersize': 4,
        'axes.linewidth': .65, 'xtick.major.width': .6,
        'ytick.major.width': .6, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'savefig.bbox': 'tight', 'figure.dpi': 100,
    })

def label(model):
    return {'original':'Original', 'retrained':'Retrained', 'finetuned':'Fine-tuned'}.get(str(model),str(model).replace('_',' ').title())

def load_metrics(root):
    paths = [root/f'metrics_{m}.csv' for m in ORDER]
    frames=[]
    for model,p in zip(ORDER,paths):
        if p.exists() and p.stat().st_size:
            x=pd.read_csv(p);x['model']=model;frames.append(x)
    p=root/'per_clip_metrics.csv'
    if p.exists():
        x=pd.read_csv(p)
        if 'model' not in x.columns: raise ValueError('per_clip_metrics.csv lacks model column')
        if not frames:frames=[x]
        else:logging.warning('Both Streamlit and CLI metrics found; using Streamlit metrics_*.csv')
    if not frames:return pd.DataFrame()
    x=pd.concat(frames,ignore_index=True)
    if 'sample_id' not in x.columns:
        for alt in ['id','clip_id']:
            if alt in x.columns:x=x.rename(columns={alt:'sample_id'});break
    if 'sample_id' not in x.columns:
        logging.warning('No sample_id; paired tests will be skipped')
    if 'split' not in x.columns:x['split']='test'  # Streamlit metrics files evaluate held-out test samples
    for col in LABELS:
        if col in x:x[col]=pd.to_numeric(x[col],errors='coerce')
    return x

def as_table(df, stem, folder, fmt=None):
    if df is None or df.empty:return
    # Preserve exact numeric values in CSV; make a clean LaTeX booktabs table.
    df.to_csv(folder/f'{stem}.csv',index=False,float_format='%.8g')
    formatted=df.copy()
    for c in formatted.select_dtypes(include=[np.number]).columns:
        formatted[c]=formatted[c].map(lambda v:'--' if pd.isna(v) else (str(int(v)) if float(v).is_integer() else f'{v:.3f}'))
    tex=formatted.to_latex(index=False,escape=True,na_rep='--',column_format=fmt,
                            caption=None,label=None,bold_rows=False)
    (folder/f'{stem}.tex').write_text('% IEEE table body (add caption and label in manuscript)\n'+tex,encoding='utf-8')

def save(fig, folder, stem, dpi):
    for ext in ['pdf','eps','png']:
        fig.savefig(folder/f'{stem}.{ext}',dpi=dpi,facecolor='white')
    plt.close(fig)

def bootstrap_ci(values, rng, nboot, alpha=.05):
    a=np.asarray(values,dtype=float);a=a[np.isfinite(a)]
    if len(a)==0:return np.nan,np.nan
    if len(a)==1:return a[0],a[0]
    # Limit matrix memory for very large datasets.
    means=[]
    for start in range(0,nboot,200):
        k=min(200,nboot-start)
        means.extend(a[rng.integers(0,len(a),(k,len(a)))].mean(axis=1))
    return map(float,np.quantile(means,[alpha/2,1-alpha/2]))

def speech_tables(x,tables,boot,seed):
    rng=np.random.default_rng(seed);rows=[]
    for (split,model),group in x.groupby(['split','model'],sort=False):
        for m in LABELS:
            if m not in group:continue
            z=group[m].dropna().to_numpy(dtype=float)
            if not len(z):continue
            lo,hi=bootstrap_ci(z,rng,boot)
            rows.append(dict(split=split,model=label(model),metric=m,n=len(z),mean=np.mean(z),std=np.std(z,ddof=1) if len(z)>1 else np.nan,ci_low=lo,ci_high=hi))
    out=pd.DataFrame(rows)
    if not out.empty:
        as_table(out,'table_speech_summary_long',tables)
        test=out[out['split']=='test'] if 'test' in out['split'].values else out
        compact=test.pivot(index='model',columns='metric',values='mean').reset_index()
        cols=['model']+[m for m in LABELS if m in compact]
        as_table(compact[cols],'table_speech_performance',tables)
    return out

def paired_table(x,tables,boot,seed):
    if 'sample_id' not in x or x.empty:return pd.DataFrame()
    rng=np.random.default_rng(seed+1);rows=[]
    for (split,metric) in [(s,m) for s in x['split'].dropna().unique() for m in LABELS if m in x]:
        t=x[x['split']==split][['sample_id','model',metric]].copy().dropna()
        t=t.groupby(['sample_id','model'],as_index=False)[metric].mean()
        p=t.pivot(index='sample_id',columns='model',values=metric)
        if 'original' not in p:continue
        for m in ['retrained','finetuned']:
            if m not in p:continue
            diff=(p[m]-p['original']).dropna().to_numpy(dtype=float)
            if not len(diff):continue
            lo,hi=bootstrap_ci(diff,rng,boot)
            rows.append({'split':split,'metric':metric,'comparison':f'{label(m)} - Original',
                         'paired_n':len(diff),'mean_delta':np.mean(diff),'ci_low':lo,'ci_high':hi})
    out=pd.DataFrame(rows);as_table(out,'table_paired_model_differences',tables)
    return out

def fig_training(root,figdir,dpi):
    path=root/'training_history.csv'
    if not path.exists():return False
    x=pd.read_csv(path)
    if not {'model','epoch'}.issubset(x):return False
    cols=[c for c in ['train_l1','val_l1'] if c in x]
    if not cols:return False
    fig,ax=plt.subplots(figsize=(3.5,2.3))
    for i,(name,g) in enumerate(x.groupby('model',sort=False)):
        g=g.sort_values('epoch')
        for j,col in enumerate(cols):
            ax.plot(g['epoch'],pd.to_numeric(g[col],errors='coerce'),
                    marker=MARKERS[i%len(MARKERS)],markevery=max(1,len(g)//10),
                    linestyle=STYLES[j],label=f'{label(name)} {col.replace("_l1","")}')
    ax.set(xlabel='Epoch',ylabel='L1 loss');ax.grid(alpha=.2,linewidth=.4)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True));ax.legend(frameon=False,fontsize=6)
    save(fig,figdir,'fig_training_curves',dpi);return True

def fig_comparison(x,figdir,dpi):
    available=[m for m in ['si_snri','output_stoi','output_estoi','output_pesq_wb','snri','output_rmse'] if m in x and x[m].notna().any()]
    if not available:return []
    split='test' if (x['split']=='test').any() else str(x['split'].iloc[0]);g=x[x['split']==split]
    models=[m for m in ORDER if (g['model']==m).any()]+[m for m in g['model'].unique() if m not in ORDER]
    made=[]
    for met in available:
        means=[g.loc[g.model==m,met].mean() for m in models]
        fig,ax=plt.subplots(figsize=(3.5,2.3))
        bars=ax.bar(range(len(models)),means,edgecolor='black',linewidth=.7,color=['0.84','0.65','0.47'][:len(models)],
                    hatch=['','//','xx'][:len(models)])
        ax.set_xticks(range(len(models)),[label(m) for m in models]);y,unit,_=LABELS[met]
        ax.set_ylabel(f'{y} ({unit})' if unit else y)
        ax.axhline(0,color='0.25',linewidth=.6)
        ax.grid(axis='y',alpha=.2,linewidth=.4)
        fig.tight_layout();stem='fig_'+met+'_comparison';save(fig,figdir,stem,dpi);made.append(stem)
    return made

def fig_cdf(x,figdir,dpi):
    if 'si_snri' not in x:return False
    g=x[x['split']=='test'] if (x['split']=='test').any() else x
    fig,ax=plt.subplots(figsize=(3.5,2.35));shown=0
    for i,(m,sub) in enumerate(g.groupby('model',sort=False)):
        vals=np.sort(sub['si_snri'].dropna().to_numpy(dtype=float))
        if not len(vals):continue
        ax.step(vals,np.arange(1,len(vals)+1)/len(vals),where='post',
                color=str(max(.15,.8-.2*i)),linestyle=STYLES[i%4],label=label(m));shown+=1
    if not shown:plt.close(fig);return False
    ax.set(xlabel='SI-SNR improvement (dB)',ylabel='Empirical CDF');ax.set_ylim(0,1.02)
    ax.grid(alpha=.2,linewidth=.4);ax.legend(frameon=False)
    save(fig,figdir,'fig_si_snri_ecdf',dpi);return True

def json_or_none(p):
    try:return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None
    except (ValueError,UnicodeError) as exc:logging.warning('Skipping unreadable %s: %s',p.name,exc);return None

def receipt_tables(root,tables,figdir,dpi):
    rows=json_or_none(root/'revocation_receipts.json')
    if rows is None:rows=json_or_none(root/'key_revocation_receipts.json')
    if not isinstance(rows,list):return
    # Export aggregated counts only, not sample identifiers or paths.
    fields=['key_absent_in_active_vault','decrypt_blocked_by_missing_key','ciphertext_retained']
    r=[]
    for k in fields:
        vals=[v[k] for v in rows if isinstance(v,dict) and isinstance(v.get(k),bool)]
        if vals:r.append({'check':k,'pass_count':sum(vals),'fail_count':len(vals)-sum(vals),'observations':len(vals)})
    z=pd.DataFrame(r);as_table(z,'table_local_key_revocation',tables)
    if z.empty:return
    fig,ax=plt.subplots(figsize=(3.5,2.35));xp=np.arange(len(z))
    ax.bar(xp,z['pass_count'],label='Pass',edgecolor='black',color='.72',linewidth=.6)
    ax.bar(xp,z['fail_count'],bottom=z['pass_count'],label='Fail',edgecolor='black',color='.3',linewidth=.6,hatch='//')
    ax.set_xticks(xp,[v.replace('_','\n') for v in z['check']],fontsize=5.7)
    ax.set_ylabel('Count');ax.yaxis.set_major_locator(MaxNLocator(integer=True));ax.legend(frameon=False)
    save(fig,figdir,'fig_local_revocation_checks',dpi)

def auxiliary_tables(root,tables,figdir,dpi):
    p=root/'unlearning_privacy_metrics.csv'
    if p.exists():
        x=pd.read_csv(p);as_table(x,'table_membership_proxy',tables)
        if 'membership_proxy_auc' in x:
            y=x[['model','membership_proxy_auc']].dropna()
            if not y.empty:
                fig,ax=plt.subplots(figsize=(3.5,2.3))
                ax.bar(np.arange(len(y)),y['membership_proxy_auc'],color='.7',edgecolor='black',hatch='//')
                ax.axhline(.5,color='black',linestyle='--',label='Chance (0.5)')
                ax.set_xticks(range(len(y)),[label(m) for m in y.model]);ax.set_ylabel('Retained-vs-test loss proxy AUC')
                ax.set_ylim(0,1);ax.legend(frameon=False)
                save(fig,figdir,'fig_membership_proxy_auc',dpi)
    audit=json_or_none(root/'privacy_audit_summary.json')
    if isinstance(audit,dict) and isinstance(audit.get('checks'),list):
        checks=pd.DataFrame([{'check':r.get('check',''),'passed':bool(r.get('pass',False)),
                              'detail':str(r.get('detail',''))[:100]} for r in audit['checks']])
        as_table(checks,'table_privacy_audit_checks',tables)
    experiment=json_or_none(root/'experiment.json')
    if isinstance(experiment,dict):
        summary={'project_setting':list(experiment.get('settings',{}).keys()),'value':[str(v) for v in experiment.get('settings',{}).values()]}
        # Exclude filesystem paths and key locations from public report.
        safe={k:v for k,v in experiment.get('settings',{}).items() if 'dir' not in k and 'root' not in k and 'path' not in k}
        if safe:as_table(pd.DataFrame({'setting':list(safe),'value':list(safe.values())}),'table_experiment_configuration',tables)

def sha256(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def run(args):
    root=Path(args.run_dir).expanduser().resolve()
    if not root.is_dir():raise FileNotFoundError(root)
    out=Path(args.output).expanduser().resolve() if args.output else root/'paper_report'
    if out==root:raise ValueError('Output must differ from experiment root')
    figdir=out/'figures';tables=out/'tables';logs=out/'logs'
    for p in [figdir,tables,logs]:p.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(logs/'report.log',encoding='utf-8'),logging.StreamHandler(sys.stdout)],force=True)
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    config();logging.info('Read-only source: %s',root)
    x=load_metrics(root)
    if x.empty:logging.warning('No per-clip metrics: speech tables and plots skipped')
    else:
        logging.info('Loaded %d metric rows, models=%s',len(x),sorted(x.model.unique()))
        speech_tables(x,tables,args.bootstrap,args.seed)
        paired_table(x,tables,args.bootstrap,args.seed)
        fig_comparison(x,figdir,args.dpi)
        fig_cdf(x,figdir,args.dpi)
    fig_training(root,figdir,args.dpi)
    receipt_tables(root,tables,figdir,args.dpi)
    auxiliary_tables(root,tables,figdir,args.dpi)
    products=sorted([p for p in out.rglob('*') if p.is_file() and p.name!='evidence_manifest.json'])
    manifest={'schema':'avse-paper-report-v1','generated_utc':datetime.now(timezone.utc).isoformat(),
              'source_type':'local experiment outputs; no source media accessed',
              'metric_rows':len(x), 'source_files':sorted([p.name for p in root.iterdir() if p.is_file() and p.suffix in ('.csv','.json')]),
              'outputs':{str(p.relative_to(out)):{'sha256':sha256(p),'bytes':p.stat().st_size} for p in products},
              'limits':['No title/caption in figures; add captions in manuscript.',
                        'Local key revocation does not prove destruction of unknown copies.',
                        'Retained-vs-test membership proxy is not deleted-sample forgetting certification.',
                        'No values are imputed for unavailable metrics.']}
    (out/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    logging.info('Generated %s',out)
    return out

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',required=True,help='Experiment folder (Streamlit or CLI)')
    p.add_argument('--output',default=None,help='Optional report directory (default: <run-dir>/paper_report)')
    p.add_argument('--dpi',type=int,default=600)
    p.add_argument('--bootstrap',type=int,default=1000)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args(argv)
    if args.bootstrap<100: p.error('--bootstrap must be >=100')
    if args.dpi<72:p.error('--dpi must be >=72')
    try:run(args)
    except Exception as exc:
        logging.exception('Reporting failed: %s',exc)
        return 1
    return 0

if __name__=='__main__':raise SystemExit(main())
