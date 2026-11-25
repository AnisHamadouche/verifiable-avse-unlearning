"""Fast synthetic test for the publication reporter. No LRS3 or keys required."""
import json
import subprocess
import sys
from pathlib import Path
import pandas as pd

SCRIPT=Path(__file__).resolve().parents[1]/'scripts'/'ieee_paper_report.py'

def test_report_streamlit_outputs(tmp_path):
    rows=[{'sample_id':f's{i}','output_stoi':0.5+i*.01,'output_estoi':0.35+i*.01,
           'si_snri':i*.2,'output_rmse':.2-i*.01} for i in range(4)]
    for model,shift in [('original',0),('retrained',-.03),('finetuned',.01)]:
        x=pd.DataFrame(rows)
        x['output_stoi']+=shift
        x.to_csv(tmp_path/f'metrics_{model}.csv',index=False)
    pd.DataFrame([{'model':'original','epoch':1,'train_l1':.32,'val_l1':.34},
                  {'model':'original','epoch':2,'train_l1':.25,'val_l1':.29}]).to_csv(tmp_path/'training_history.csv',index=False)
    (tmp_path/'revocation_receipts.json').write_text(json.dumps([
        {'sample_id':'withdrawn_test','key_absent_in_active_vault':True,
         'decrypt_blocked_by_missing_key':True,'ciphertext_retained':True}]))
    out=tmp_path/'report'
    p=subprocess.run([sys.executable,str(SCRIPT),'--run-dir',str(tmp_path),'--output',str(out),
                      '--bootstrap','100','--dpi','90'],capture_output=True,text=True)
    assert p.returncode==0,p.stdout+p.stderr
    assert (out/'tables/table_speech_performance.tex').exists()
    assert (out/'tables/table_paired_model_differences.csv').exists()
    for ext in ['png','eps','pdf']:
        assert (out/f'figures/fig_output_stoi_comparison.{ext}').exists()
    assert (out/'logs/report.log').exists()
    manifest=json.loads((out/'evidence_manifest.json').read_text())
    assert manifest['metric_rows']==12
    assert all('sample_id' not in p.name for p in (out/'tables').iterdir())

def test_report_original_cli_outputs(tmp_path):
    pd.DataFrame([{'model':'original','split':'test','sample_id':'a','si_snri':1.0},
                  {'model':'retrained','split':'test','sample_id':'a','si_snri':.9}]).to_csv(tmp_path/'per_clip_metrics.csv',index=False)
    p=subprocess.run([sys.executable,str(SCRIPT),'--run-dir',str(tmp_path),
                      '--bootstrap','100','--dpi','72'],capture_output=True,text=True)
    assert p.returncode==0,p.stdout+p.stderr
    assert (tmp_path/'paper_report/tables/table_speech_performance.tex').exists()
