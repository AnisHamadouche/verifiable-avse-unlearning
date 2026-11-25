"""Prevent tracking obviously sensitive recordings, secrets and model checkpoints."""
import subprocess,sys
from pathlib import Path
files=subprocess.check_output(['git','ls-files'],text=True).splitlines()
forbidden_suffixes={'.mp4','.flac','.wav','.key','.pem','.p12','.pfx','.pt','.pth','.ckpt','.enc','.npz'}
forbidden_dirs={'app_data','recordings','encrypted_cache','runs','private_keys','.venv','LRS3-TED'}
errors=[f for f in files if Path(f).suffix.lower() in forbidden_suffixes or bool(forbidden_dirs & set(Path(f).parts))]
if errors:
    print('Refusing to publish potentially sensitive/generated files:',*errors,sep='\n')
    sys.exit(1)
print('PASS: no prohibited tracked paths')
