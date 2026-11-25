"""Local AVSE privacy research dashboard. Not a medical or certified-erasure tool."""
from __future__ import annotations
import io
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch

from privacy_backend import Lab, image_to_lips, decode_mp4_bytes, verify_certificate
from avse_unlearning import SR, NSAMPLES, NFRAMES, VWIDTH
from playback import audio_wav_bytes, lip_preview_mp4
from audio_preview import make_preview_audio

st.set_page_config(page_title='AVSE Privacy Lab',page_icon='🔐',layout='wide')
st.title('AVSE Privacy Lab · Capture → Encrypt → Withdraw → Unlearn → Audit')
st.caption('Local research prototype | Audio-visual enhancement · AES-256-GCM · selective withdrawal · retraining reference')
st.warning('Research demonstration only. A signed local receipt is NOT proof of universal deletion or certified machine forgetting.',icon='⚠️')

class CaptureBuffer:
    def __init__(self):
        self.lock=threading.Lock();self.audio=[];self.lips=[];self.found=0
        self.last_frame=0.;self.audio_start=None;self._resampler=None

    def on_audio(self,frame):
        import av
        if self._resampler is None:
            self._resampler=av.AudioResampler(format='s16',layout='mono',rate=SR)
        for chunk in self._resampler.resample(frame):
            samples=chunk.to_ndarray().reshape(-1).astype(np.float32)/32768.
            with self.lock:
                if self.audio_start is None:self.audio_start=time.monotonic()
                if sum(len(a) for a in self.audio)<SR*30:self.audio.append(samples)
        return frame

    def on_video(self,frame):
        now=time.monotonic()
        if now-self.last_frame<0.095:return frame
        self.last_frame=now
        img=frame.to_ndarray(format='bgr24')
        lip,found=image_to_lips(img)
        with self.lock:
            if len(self.lips)<300:
                self.lips.append((now,lip));self.found+=int(found)
        return frame

    def sync_report(self):
        with self.lock:
            audio_count=sum(len(x) for x in self.audio)
            frame_times=[float(t) for t,_ in self.lips]
            start=self.audio_start
            found=self.found
        return {'audio_samples_captured':audio_count,
                'video_frames_captured':len(frame_times),
                'face_detected_fraction':round(found/max(len(frame_times),1),3),
                'estimated_capture_span_s':round(max(frame_times)-min(frame_times),3) if len(frame_times)>1 else 0.,
                'sync_method':'Approximate monotonic callbacks; not hardware timestamp synchronisation'}

    def export(self):
        with self.lock:
            if not self.audio or not self.lips:raise ValueError('Record with both mic and camera enabled')
            x=np.concatenate(self.audio)
            times=np.array([t for t,_ in self.lips])
            frames=[im.copy() for _,im in self.lips]
            start=float(self.audio_start)
        if len(x)<SR//2:raise ValueError('Please record at least 1–2 seconds of speech')
        target=start+np.arange(NFRAMES)*(NSAMPLES/SR)/NFRAMES
        idx=np.abs(times[:,None]-target[None,:]).argmin(axis=0)
        return x[:NSAMPLES],np.stack([frames[int(i)] for i in idx])


def read_audio_input(blob):
    import soundfile as sf
    x,sr=sf.read(io.BytesIO(blob),dtype='float32')
    if x.ndim>1:x=x.mean(axis=1)
    if sr!=SR:
        from scipy.signal import resample_poly
        import math
        g=math.gcd(sr,SR);x=resample_poly(x,SR//g,sr//g).astype('float32')
    return x

root=Path(st.sidebar.text_input('Experiment directory',value=str(Path.cwd()/'app_data'/'demo'))).expanduser()
key_dir=Path(st.sidebar.text_input('Separate key directory',value=str(Path.home()/'.local/share/avse-streamlit-keys'))).expanduser()
lab=Lab(root,key_dir)
st.sidebar.metric('CUDA', 'Available' if torch.cuda.is_available() else 'CPU only')
st.sidebar.write('Data and keys remain on the machine running Streamlit. Use localhost for private recordings.')
st.sidebar.caption('Storage: AES-GCM encrypted derived clips; private keys in separate directory; receipts are operator-issued.')

PAGES = ['1 · Capture & import', '2 · Playback & preview', '3 · Train original / unlearn',
         '4 · Records & deletion', '5 · Metrics', '6 · Certificates', '7 · Assurance & provenance', '8 · Automated privacy audit']
if 'page_index' not in st.session_state:
    st.session_state.page_index = 0

def go_to_page(i):
    st.session_state.page_index = max(0, min(i, len(PAGES) - 1))
    st.session_state.navigation_select = PAGES[st.session_state.page_index]

st.sidebar.divider()
st.sidebar.subheader('Experiment workflow')
st.sidebar.radio('Go to page', PAGES, key='navigation_select',
                 index=st.session_state.page_index,
                 on_change=lambda: go_to_page(PAGES.index(st.session_state.navigation_select)))
current_page = st.session_state.page_index
st.progress((current_page + 1) / len(PAGES))
st.caption(f'Page {current_page + 1} of {len(PAGES)} — {PAGES[current_page]}')
left_nav, mid_nav, right_nav = st.columns([1, 5, 1])
with left_nav:
    st.button('← Back', key='top_back', disabled=current_page == 0,
              on_click=go_to_page, args=(current_page - 1,))
with right_nav:
    st.button('Next →', key='top_next', disabled=current_page == len(PAGES)-1,
              on_click=go_to_page, args=(current_page + 1,))


if current_page == 0:
    st.subheader('Record synchronised microphone and camera streams')
    st.write('Start the live stream, speak while facing the camera, stop it, then save a two-second clip. The face detector extracts an **approximate lower-face mouth crop**, not precise lip landmarks.')
    if 'capture_buffer' not in st.session_state:st.session_state.capture_buffer=CaptureBuffer()
    try:
        from streamlit_webrtc import webrtc_streamer,WebRtcMode
        ctx=webrtc_streamer(key='avse-stream',mode=WebRtcMode.SENDONLY,
                           video_frame_callback=st.session_state.capture_buffer.on_video,
                           audio_frame_callback=st.session_state.capture_buffer.on_audio,
                           media_stream_constraints={'video':True,'audio':True},
                           async_processing=True)
        st.caption('Browser camera/microphone permission required. For remote access, HTTPS and proper TURN/STUN setup may be necessary.')
        with st.form('save_stream'):
            label=st.text_input('Clip label',value='volunteer_sample')
            split=st.selectbox('Use for','train test'.split())
            submitted=st.form_submit_button('Save encrypted stream clip',disabled=bool(ctx.state.playing))
        if submitted:
            try:
                quality=st.session_state.capture_buffer.sync_report()
                clean,lips=st.session_state.capture_buffer.export()
                row=lab.add(clean,lips,label,split,'webrtc')
                lab.event('capture_quality',sample_id=row['id'],**quality)
                st.json(quality)
                st.success(f"Saved encrypted {row['id']} ({split}); sample key stored separately")
                st.image(lips[0],caption='Extracted mouth region (approximate)',width=135)
                st.session_state.capture_buffer=CaptureBuffer()
            except Exception as e: st.error(str(e))
    except ImportError:
        st.error('Install streamlit-webrtc and av to capture live streams.')

    st.divider()
    st.subheader('Alternative: import an MP4 clip')
    mp4=st.file_uploader('MP4 containing video and audio',type=['mp4'],key='mp4')
    imp_label=st.text_input('Imported clip label',value='uploaded_clip')
    imp_split=st.selectbox('Split',('train','test'),key='imp_split')
    if st.button('Extract lips + encrypt imported MP4',disabled=mp4 is None):
        try:
            audio,lips=decode_mp4_bytes(mp4.getvalue())
            row=lab.add(audio,lips,imp_label,imp_split,'uploaded_mp4')
            st.success(f"Encrypted {row['id']}");st.image(lips[0],width=140)
        except Exception as exc:st.error(f'Import failed: {exc}')

    st.divider()
    with st.expander('Optional: import from local LRS3 MP4 paths'):
        st.caption('Uses a local directory path; no LRS3 media is uploaded to GitHub. Respect the dataset licence.')
        lrs=st.text_input('Local LRS3 root',value=str(Path.home()/'LRS3-TED'))
        st.caption('No upper import limit. Enter 0 to import ALL available clips in that split. Large imports can consume significant time and disk space.')
        n=st.number_input('Training clips to import (0 = all)',min_value=0,max_value=None,value=12,step=1)
        nt=st.number_input('Test clips to import (0 = all)',min_value=0,max_value=None,value=3,step=1)
        skip_existing=st.checkbox('Skip clips already imported by this importer',value=True,
                                  help='Tracks source paths for new imports. Older records without recorded paths cannot be matched reliably.')
        if st.button('Import LRS3 clips'):
            from itertools import islice
            root_lrs=Path(lrs).expanduser().resolve()
            count=0; skipped=0; failed=0; first_error=None
            existing={r['source'] for r in lab.records() if str(r.get('source','')).startswith('LRS3-local:')} if skip_existing else set()
            status=st.empty()
            progress=st.progress(0.0)
            for subset,lim,split in [('trainval',int(n),'train'),('test',int(nt),'test')]:
                split_root=root_lrs/subset
                if not split_root.is_dir():
                    st.warning(f'Missing dataset split: {split_root}')
                    continue
                # Stream paths without constructing a list of every MP4 in memory.
                paths=split_root.rglob('*.mp4')
                if lim:
                    paths=islice(paths,lim)
                examined=0
                for path in paths:
                    examined+=1
                    source='LRS3-local:'+str(path.relative_to(root_lrs))
                    if skip_existing and source in existing:
                        skipped+=1
                    else:
                        try:
                            # Read and encrypt one clip at a time (no batch plaintext cache).
                            audio,lips=decode_mp4_bytes(path.read_bytes())
                            lab.add(audio,lips,path.stem,split,source)
                            existing.add(source)
                            count+=1
                        except Exception as e:
                            failed+=1
                            if first_error is None:first_error=f'{path}: {e}'
                    if examined%10==0:
                        status.info(f'{subset}: processed {examined} files · imported {count} · skipped {skipped} · errors {failed}')
                        if lim:progress.progress(min(1.,examined/lim))
                status.info(f'{subset} complete · examined {examined} files · imported {count} total')
                progress.progress(1.0)
            st.success(f'Imported {count} encrypted clips. Skipped {skipped} existing; {failed} failed.')
            if first_error:st.warning(f'First import error: {first_error}')
    with st.expander('Generate small synthetic demonstration samples'):
        st.caption('Synthetic harmonic signals and animated mock mouth crops. Only a software smoke test; NOT a realistic speech dataset.')
        if st.button('Generate 12 train + 3 test synthetic clips'):
            rng=np.random.default_rng(42)
            for i in range(15):
                t=np.arange(NSAMPLES)/SR
                amp=.15+.08*np.sin(2*np.pi*(2+i*.06)*t)
                audio=(amp*(np.sin(2*np.pi*(140+i*9)*t)+.3*np.sin(2*np.pi*(280+i*18)*t))).astype('float32')
                frames=np.zeros((NFRAMES,VWIDTH,VWIDTH),dtype='uint8')
                for j in range(NFRAMES):
                    cv2.ellipse(frames[j],(24,24),(11,4+int(3*np.sin(j+i)**2)),0,0,360,180,-1)
                frames=np.clip(frames+rng.integers(0,12,frames.shape,dtype=np.uint8),0,255)
                lab.add(audio,frames,f'synthetic_{i}', 'train' if i<12 else 'test','synthetic')
            st.success('Synthetic clips added')

if current_page == 1:
    st.subheader('Listen to audio and watch stored lip-region frames')
    st.caption('Preview decrypts ONE selected active record in memory. The app stores mouth-region crops, not full-face video. Revoked records cannot be played.')
    inventory=lab.records()
    playable=[r for r in inventory if r['status']=='active']
    if not playable:
        st.info('No active recordings available for playback. Record or import a clip first.')
    else:
        st.dataframe(pd.DataFrame(inventory)[['id','label','split','status','source']], use_container_width=True)
        options={f"{r['label']} · {r['split']} · {r['id']}":r['id'] for r in playable}
        selected=st.selectbox('Select a recording to play', list(options), key='playback_choice')
        sid=options[selected]
        try:
            # Never retain decrypted recordings in session_state or use st.cache_data.
            audio, lips = lab.vault.load(sid)
            st.write(f'Audio: **{len(audio) / SR:.2f} s**, {SR:,} Hz mono; lip frames: **{len(lips)}**, 48 × 48 pixels.')
            st.subheader('Audio comparison')
            st.caption('Noisy = 0 dB two-speaker babble from currently active training clips. Audio is generated in memory; no plaintext files are saved.')
            checkpoints = {'Original baseline': lab.root / 'original.pt',
                           'Retrained without withdrawn records': lab.root / 'retrained_without_deleted.pt',
                           'Fine-tuned model': lab.root / 'finetuned_without_deleted.pt'}
            revoked_present=any(r['status']=='revoked' for r in lab.records())
            available_models = {k: v for k, v in checkpoints.items()
                                if v.exists() and not (revoked_present and k=='Original baseline')}
            model_choice = st.selectbox('Enhancement model', list(available_models) if available_models else ['None trained'], key='preview_model')
            try:
                clean, noisy, enhanced = make_preview_audio(lab, sid, available_models.get(model_choice), seed=42, snr=0.0)
                c_original, c_noisy, c_cleaned = st.columns(3)
                with c_original:
                    st.markdown('**Original clean audio**')
                    st.audio(audio_wav_bytes(clean, SR), format='audio/wav')
                with c_noisy:
                    st.markdown('**Noisy audio (babble)**')
                    st.audio(audio_wav_bytes(noisy, SR), format='audio/wav')
                with c_cleaned:
                    st.markdown('**Enhanced / cleaned audio**')
                    if enhanced is None:
                        st.info('Train the original baseline to enable enhanced audio.')
                    else:
                        st.audio(audio_wav_bytes(enhanced, SR), format='audio/wav')
                st.caption('Enhancement uses the selected saved model, not the clean target. Preview noise is reproducibly sampled from currently accessible clips; after deletion, it can differ from noise used in earlier evaluations.')
            except (ValueError, RuntimeError, FileNotFoundError) as exc:
                st.warning(f'Noise/enhancement preview unavailable: {exc}')
                st.audio(audio_wav_bytes(audio, SR), format='audio/wav')
            st.subheader('Synchronized lip video')
            st.caption('The video includes the stored lip crops and ORIGINAL audio; it is not the enhanced output.')
            try:
                st.video(lip_preview_mp4(audio, lips, sample_rate=SR), format='video/mp4')
            except (RuntimeError, FileNotFoundError, OSError) as ex:
                st.warning(f'Combined playback unavailable ({ex}). Lip-frame gallery below remains available. Install ffmpeg for combined playback.')
            with st.expander('Inspect individual lip frames'):
                frame_index=st.slider('Lip frame', 0, len(lips)-1, 0, key='lip_frame')
                st.image(lips[frame_index], caption=f'Frame {frame_index+1}/{len(lips)} — mouth crop only', width=240, clamp=True)
            st.info('Preview data are temporarily decrypted in server/browser memory. Do not use this app on a public server with private recordings.')
        except FileNotFoundError:
            st.error('The selected sample is inaccessible because its decryption key is missing or was revoked.')
        except Exception as ex:
            st.error(f'Unable to load this recording: {ex}')
    deleted=[r for r in inventory if r['status']=='revoked']
    if deleted:
        st.caption(f'{len(deleted)} revoked recording(s) are intentionally hidden from playback.')

if current_page == 3:
    st.subheader('Encrypted record inventory')
    rows=lab.records()
    if rows:
        st.dataframe(pd.DataFrame(rows)[['id','label','split','status','source','created']],use_container_width=True)
        active_train=[r for r in rows if r['status']=='active' and r['split']=='train']
        options={f"{r['label']} · {r['id']}":r['id'] for r in active_train}
        choices=st.multiselect('Select TRAIN records to withdraw (not the whole set)',list(options))
        st.info('Deletion removes each selected encryption key from this local key directory. Encrypted ciphertext may remain. Original source recordings and model checkpoints are not deleted.')
        if st.button('Request selective deletion / revoke keys',type='primary'):
            try:
                out=lab.delete([options[x] for x in choices]);st.success(f"Revoked {len(out['receipts'])} selected keys")
                st.json(out)
            except Exception as e:st.error(str(e))
    else:st.info('Record or import clips to begin.')

if current_page == 2:
    st.subheader('AVSE training and unlearning')
    st.write('Uses the same compact PyTorch audio-STFT + visual-CNN + GRU model as the LRS3 experiment. A fixed training/test split is constructed from encrypted clips.')
    epochs=st.number_input('Epochs',min_value=1,max_value=200,value=3)
    batch=st.number_input('Batch size',min_value=1,max_value=32,value=4)
    architecture=st.selectbox('AVSE model', ['tiny','cross_attention'],
                              help='Cross-attention is a research baseline; requires more memory and training.')
    if st.button('Train original AVSE baseline',type='primary'):
        try:
            with st.spinner('Training original model and evaluating test clips...'):
                hist=lab.train_original(epochs,batch,architecture=architecture)
            st.success(f'Training completed: {len(hist)} epoch records')
        except Exception as e:st.error(str(e))
    st.divider()
    approx=st.checkbox('Also run approximate fine-tuning baseline',value=True)
    st.write('**Reference unlearning:** fresh retraining on retained records. **Approximate unlearning:** fine-tuning contaminated original weights; not guaranteed to forget.')
    if st.button('Apply machine unlearning after deletion'):
        try:
            with st.spinner('Retraining and evaluating retained samples...'):
                out=lab.unlearn(epochs,batch,approx)
            st.success('Unlearning comparisons completed');st.json(out)
        except Exception as e:st.error(str(e))

if current_page == 4:
    st.subheader('Model metrics on the untouched test split')
    available=[]
    for name in ['original','retrained','finetuned']:
        file=lab.root/f'metrics_{name}.csv'
        if file.exists():
            df=pd.read_csv(file);df['model']=name;available.append(df)
    if available:
        data = pd.concat(available, ignore_index=True)
        columns = [c for c in [
            'input_si_snr','output_si_snr','si_snri',
            'input_snr','output_snr','snri',
            'input_segmental_snr','output_segmental_snr',
            'input_rmse','output_rmse','input_mae','output_mae',
            'input_stoi','output_stoi','stoi_improvement',
            'input_estoi','output_estoi','estoi_improvement',
            'input_pesq_wb','output_pesq_wb','pesq_wb_improvement'
        ] if c in data.columns]
        if columns:
            labels = {'input_si_snr':'Input SI-SNR (dB)', 'output_si_snr':'Output SI-SNR (dB)',
                      'si_snri':'SI-SNRi (dB)', 'input_snr':'Input SNR (dB)',
                      'output_snr':'Output SNR (dB)', 'snri':'SNRi (dB)',
                      'input_segmental_snr':'Input segSNR (dB)', 'output_segmental_snr':'Output segSNR (dB)',
                      'input_rmse':'Input RMSE', 'output_rmse':'Output RMSE',
                      'input_mae':'Input MAE', 'output_mae':'Output MAE',
                      'input_stoi':'Input STOI', 'output_stoi':'Output STOI',
                      'stoi_improvement':'STOI Δ', 'input_estoi':'Input ESTOI',
                      'output_estoi':'Output ESTOI', 'estoi_improvement':'ESTOI Δ',
                      'input_pesq_wb':'Input PESQ-WB', 'output_pesq_wb':'Output PESQ-WB',
                      'pesq_wb_improvement':'PESQ-WB Δ'}
            st.markdown('### Select metrics to display')
            st.caption('Tick the metrics you want to see. Your choices apply to the summary table, chart and individual-clip results.')
            groups = {
                'Intelligibility (STOI / ESTOI)': [
                    'input_stoi', 'output_stoi', 'stoi_improvement',
                    'input_estoi', 'output_estoi', 'estoi_improvement'],
                'Speech quality (PESQ-WB)': [
                    'input_pesq_wb', 'output_pesq_wb', 'pesq_wb_improvement'],
                'Enhancement (SI-SNR / SNR)': [
                    'input_si_snr', 'output_si_snr', 'si_snri',
                    'input_snr', 'output_snr', 'snri',
                    'input_segmental_snr', 'output_segmental_snr'],
                'Reconstruction error (RMSE / MAE)': [
                    'input_rmse', 'output_rmse', 'input_mae', 'output_mae'],
            }
            default_metrics = {
                'output_stoi', 'output_estoi', 'si_snri', 'snri',
                'output_pesq_wb',
            }
            c_select, c_clear = st.columns([1, 1])
            with c_select:
                if st.button('Select all available metrics', key='metrics_select_all'):
                    for metric in columns:
                        st.session_state[f'metric_selected_{metric}'] = True
            with c_clear:
                if st.button('Clear metric selection', key='metrics_clear_all'):
                    for metric in columns:
                        st.session_state[f'metric_selected_{metric}'] = False
            selected = []
            for title, group in groups.items():
                available_group = [metric for metric in group if metric in columns]
                if not available_group:
                    continue
                with st.expander(title, expanded=(title.startswith('Intelligibility') or title.startswith('Enhancement'))):
                    cells = st.columns(3)
                    for i, metric in enumerate(available_group):
                        with cells[i % 3]:
                            if st.checkbox(labels.get(metric, metric), value=metric in default_metrics,
                                           key=f'metric_selected_{metric}'):
                                selected.append(metric)
            st.caption(f'{len(selected)} of {len(columns)} metrics selected')
            if selected:
                summary = data.groupby('model')[selected].agg(['mean','std','count'])
                summary.columns = [f'{labels.get(a,a)} · {b}' for a,b in summary.columns]
                st.markdown('**Selected metrics — mean, standard deviation and sample count**')
                st.dataframe(summary.round(4), use_container_width=True)
                st.download_button('Download selected summary CSV', summary.to_csv().encode(),
                                   file_name='avse_metrics_selected_summary.csv', mime='text/csv')
                selected_metric = st.selectbox('Metric to chart', selected,
                                               index=0,
                                               format_func=lambda x: labels.get(x, x))
                st.bar_chart(data.groupby('model')[selected_metric].mean())
                st.markdown('**Individual clip results — selected metrics**')
                id_columns = [c for c in ['sample_id', 'model', 'split'] if c in data.columns]
                shown_data = data[id_columns + selected]
                st.dataframe(shown_data.round(4), use_container_width=True)
                st.download_button('Download selected per-clip metrics CSV',
                                   shown_data.to_csv(index=False).encode(),
                                   file_name='avse_metrics_selected_per_clip.csv', mime='text/csv')
                if 'original' in data.model.values and 'retrained' in data.model.values:
                    subset = data[data.model.isin(['original','retrained'])]
                    if 'sample_id' in subset.columns:
                        d = subset.pivot_table(index='sample_id', columns='model', values=selected_metric)
                        if {'original','retrained'}.issubset(d.columns):
                            paired = (d['retrained']-d['original']).dropna()
                            st.metric('Retained-data retraining vs original (mean paired difference)',
                                      f'{paired.mean():+.4f}' if len(paired) else 'N/A')
            else:
                st.info('Tick at least one metric above to display comparisons and download selected results.')
        else:
            st.warning('Metrics file lacks recognised columns')
        st.caption('STOI/ESTOI = intelligibility estimates; PESQ-WB = quality; SI-SNRi/SNRi = enhancement. None establishes machine forgetting. PESQ is shown only if computed.')
    else:
        st.info('Run baseline training to produce metrics.')
    st.divider()
    st.subheader('Machine-unlearning audit (empirical proxy)')
    privacy_path=lab.root/'unlearning_privacy_metrics.csv'
    if privacy_path.exists():
        pm=pd.read_csv(privacy_path)
        st.dataframe(pm.round(4),use_container_width=True)
        audit_columns=[c for c in ['membership_proxy_auc','retained_mean_l1','heldout_mean_l1','audit_seconds'] if c in pm.columns]
        if audit_columns:
            chosen_privacy=st.multiselect('Select privacy-audit measures',audit_columns,default=audit_columns)
            if chosen_privacy:st.bar_chart(pm.set_index('model')[chosen_privacy])
        st.download_button('Download privacy-audit CSV',privacy_path.read_bytes(),file_name='unlearning_privacy_metrics.csv')
        st.warning('Membership AUC is a retained-vs-test loss-threshold proxy. Deleted-sample membership and certified forgetting are NOT measured.')
    else:st.info('Run unlearning to calculate retained-vs-test membership proxy and audit latency.')
    hist=lab.root/'training_history.csv'
    if hist.exists():
        d=pd.read_csv(hist)
        st.subheader('Loss curves')
        st.line_chart(d.pivot_table(index='epoch',columns='model',values='val_l1',aggfunc='mean'))

if current_page == 5:
    st.subheader('Local verification statements and audit certificates')
    st.write('Each deletion receipt records the ciphertext hash and that the local key store refused access. The Ed25519 signature authenticates the **operator-issued statement**, not an independently witnessed event.')
    last=lab.root/'last_deletion_receipts.json'
    if last.exists():
        records=json.loads(last.read_text())['receipts']
        st.metric('Reported key revocations',len(records))
        st.metric('Blocked local decryptions',sum(bool(r['decrypt_blocked_by_missing_key']) for r in records))
        st.dataframe(pd.DataFrame(records)[['sample_id','key_absent_in_active_vault','decrypt_blocked_by_missing_key']],use_container_width=True)
        st.download_button('Download revocation receipts',last.read_bytes(),file_name='key_revocation_receipts.json',mime='application/json')
    else:st.info('Delete at least one sample to generate receipts.')
    if st.button('Issue signed local audit certificate'):
        try:
            doc=lab.issue_certificate()
            st.success('Local statement signed by the operator-controlled Ed25519 key')
            st.json(doc)
        except Exception as e:st.error(str(e))
    cert=lab.root/'audit_certificate.json'
    if cert.exists():
        data=json.loads(cert.read_text())
        try:
            verify_certificate(data)
            st.success('Ed25519 signature valid (document integrity/authenticity only)')
        except Exception as exc:st.error(f'Invalid certificate signature: {exc}')
        st.download_button('Download certificate JSON',cert.read_bytes(),file_name='avse_audit_certificate.json',mime='application/json')
    st.divider()
    st.caption('For independent verification, download the certificate and verifier public key, pin the key outside the experiment, and check the hash-chain root against an externally retained checkpoint. This does NOT attest unknown copies or certified forgetting.')
    pinned=lab.root/'verifier_public_key.hex'
    if pinned.exists():
        st.download_button('Download public key to pin externally',pinned.read_bytes(),file_name='avse_pinned_public_key.hex',mime='text/plain')
    st.code('python verify_independent.py --certificate audit_certificate.json --pinned-public-key ~/trusted/avse_pinned_public_key.hex --chain event_chain.json',language='bash')

if current_page == 6:
    st.subheader('P1–P3: Registered lineage, withdrawal impact and verification')
    st.caption('This is a registry of *known local* artifacts, not a proof of copy completeness.')
    registry=lab.root/'lineage.json'
    if registry.exists():
        graph=json.loads(registry.read_text())
        nodes=[{'artifact':k,'kind':v.get('kind'),'dependencies':len(v.get('sources',[])),
                'hash':v.get('sha256')} for k,v in graph.get('nodes',{}).items()]
        st.dataframe(pd.DataFrame(nodes),use_container_width=True)
        st.download_button('Download lineage graph',registry.read_bytes(),file_name='lineage.json',mime='application/json')
    impacted=lab.root/'withdrawal_impact.json'
    if impacted.exists():
        inventory=json.loads(impacted.read_text())
        st.metric('Known dependent artifacts affected',len(inventory['affected_registered_artifacts']))
        st.dataframe(pd.DataFrame([{'artifact':k,'kind':v.get('kind')} for k,v in inventory['affected_registered_artifacts'].items()]))
        st.download_button('Download withdrawal impact report',impacted.read_bytes(),file_name='withdrawal_impact.json',mime='application/json')
        st.warning('Affected model checkpoints must not be served. Remaining plaintext copies or third-party exports are not observable by this prototype.')
    event_chain=lab.root/'event_chain.json'
    if event_chain.exists():
        from evidence import verify_chain
        try:
            current_root=verify_chain(json.loads(event_chain.read_text()))
            st.success('Local hash-chain consistency verified')
            st.code(current_root)
            st.download_button('Export event chain',event_chain.read_bytes(),file_name='event_chain.json')
            st.caption('Record the root OUTSIDE the operator machine before future events; otherwise truncation and rewriting can go undetected.')
        except Exception as ex:st.error(f'Hash chain inconsistent: {ex}')
    policy=lab.root/'model_serving_policy.json'
    if policy.exists():st.json(json.loads(policy.read_text()))
    st.info('Independent verification requires an externally pinned public key. A local hash chain alone is not an external witness.')


if current_page == 7:
    st.subheader('Automated healthcare privacy audit')
    st.caption('Read-only audit of registered local records, revoked keys, ciphertext hashes, model-retirement state, provenance and certificate evidence. The result is NOT a regulatory certification.')
    st.info('GitHub Actions runs this audit on synthetic records. Here you can audit the current local experiment without uploading recordings or key material.')
    st.markdown('**Verification settings**')
    require_cert = st.checkbox('Require certificate verification against an independently pinned key', value=False,
                               help='Turn on only after exporting and pinning the trusted public key outside the application.')
    pinned_path = st.text_input('Pinned public key file (verifier-controlled)', value='',
                                disabled=not require_cert)
    witnessed_root = st.text_input('Externally witnessed chain root (64-character SHA-256 hex)', value='',
                                   disabled=not require_cert)
    if st.button('Run privacy audit', type='primary'):
        from scripts.privacy_audit import audit
        import re
        try:
            if require_cert:
                if not pinned_path.strip():
                    raise ValueError('Specify a previously pinned verifier key file.')
                if not re.fullmatch(r'[0-9a-fA-F]{64}', witnessed_root.strip()):
                    raise ValueError('An independently witnessed 64-character chain root is required.')
            result = audit(lab.root, lab.key_root, require_certificate=require_cert,
                           pinned_key=Path(pinned_path).expanduser() if require_cert else None,
                           witnessed_root=witnessed_root.strip().lower() if require_cert else None)
            result_path = lab.root / 'privacy_audit_summary.json'
            result_path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
            st.session_state['current_audit_report'] = result
            if result.get('passed'):
                st.success('All checks within the selected audit scope passed.')
            else:
                st.error('Privacy audit failed. Inspect failed checks before further training or release.')
        except Exception as exc:
            st.error(f'Audit could not complete: {exc}')
    audit_result = st.session_state.get('current_audit_report')
    saved_report = lab.root / 'privacy_audit_summary.json'
    if saved_report.exists():
        audit_result = json.loads(saved_report.read_text(encoding='utf-8'))
    if audit_result:
        checks = pd.DataFrame(audit_result.get('checks', []))
        st.metric('Checks', len(checks))
        if len(checks):
            a, b = st.columns(2)
            a.metric('Passed', int(checks['pass'].sum()))
            b.metric('Failed', int((~checks['pass']).sum()))
            st.dataframe(checks, use_container_width=True)
        st.download_button('Download local privacy audit JSON',
                           json.dumps(audit_result, indent=2).encode('utf-8'),
                           file_name='privacy_audit_summary.json', mime='application/json')
        with st.expander('Audit scope and limitations', expanded=True):
            st.write(audit_result.get('scope', 'Local registered data only'))
            for note in audit_result.get('limitations', []):
                st.write('• ' + note)
    st.divider()
    st.markdown('**Run the same checks from the command line**')
    st.code('python scripts/privacy_audit.py --root app_data/demo --key-root ~/.local/share/avse-streamlit-keys --output /tmp/avse-audit.json', language='bash')
    st.caption('For independent checks, add --require-certificate --pinned-key PATH --witnessed-root HASH. Store the trust anchor and witnessed root outside the operator environment.')

st.divider()
footer_left, footer_center, footer_right = st.columns([1, 5, 1])
with footer_left:
    st.button('← Back', key='bottom_back', disabled=current_page == 0, on_click=go_to_page, args=(current_page-1,))
with footer_right:
    st.button('Next →', key='bottom_next', disabled=current_page == len(PAGES)-1, on_click=go_to_page, args=(current_page+1,))
