"""Objective speech metrics for aligned, 16-kHz clean/noisy/enhanced waveforms.

STOI/ESTOI are intrusive intelligibility estimates. PESQ-WB is quality,
not intelligibility. All assume clean reference alignment; no unlearning claim.
"""
import numpy as np


def _wave(x):
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if len(a) < 8000 or not np.all(np.isfinite(a)):
        raise ValueError("Expect >= 0.5 s of finite waveform at 16 kHz")
    return a


def calculate(clean, noisy, enhanced, sr=16000, include_pesq=False):
    if sr != 16000:
        raise ValueError("Expected 16000 Hz speech; resample before evaluation")
    try:
        from pystoi import stoi
    except ImportError as e:
        raise RuntimeError("Install pystoi: pip install pystoi") from e
    clean, noisy, enhanced = map(_wave, (clean, noisy, enhanced))
    if len(clean) != len(noisy) or len(clean) != len(enhanced):
        raise ValueError("Waveforms must be exactly sample-aligned")
    def snr_db(reference, estimate):
        return float(10*np.log10((np.mean(reference**2)+1e-10)/(np.mean((reference-estimate)**2)+1e-10)))
    def segmental_snr(reference, estimate, block=512):
        values=[]
        for pos in range(0,len(reference)-block+1,block):
            r=reference[pos:pos+block]; e=estimate[pos:pos+block]
            if np.mean(r**2) < 1e-7: continue
            values.append(float(np.clip(snr_db(r,e),-10,35)))
        return float(np.mean(values)) if values else float('nan')
    result = {
        'input_snr':snr_db(clean,noisy), 'output_snr':snr_db(clean,enhanced),
        'input_segmental_snr':segmental_snr(clean,noisy),
        'output_segmental_snr':segmental_snr(clean,enhanced),
        'input_rmse':float(np.sqrt(np.mean((clean-noisy)**2))),
        'output_rmse':float(np.sqrt(np.mean((clean-enhanced)**2))),
        'input_mae':float(np.mean(np.abs(clean-noisy))),
        'output_mae':float(np.mean(np.abs(clean-enhanced))),
    }
    result['snri'] = result['output_snr'] - result['input_snr']
    for name, extended in (("stoi", False), ("estoi", True)):
        a = float(stoi(clean, noisy, sr, extended=extended))
        b = float(stoi(clean, enhanced, sr, extended=extended))
        result[f"input_{name}"] = a
        result[f"output_{name}"] = b
        result[f"{name}_improvement"] = b - a
    if include_pesq:
        try:
            from pesq import pesq
        except ImportError as e:
            raise RuntimeError("Install optional PESQ dependency: pip install pesq") from e
        # PESQ can reject problematic waveforms; record NaN explicitly, not zero.
        for label, signal in (("input_pesq_wb", noisy), ("output_pesq_wb", enhanced)):
            try:
                result[label] = float(pesq(sr, clean.astype(np.float32),
                                           signal.astype(np.float32), "wb"))
            except Exception:
                result[label] = float("nan")
        result["pesq_wb_improvement"] = result["output_pesq_wb"] - result["input_pesq_wb"]
    return result
