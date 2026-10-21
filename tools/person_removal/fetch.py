"""Fetch a video method's code (at the commit these tools were tested with) and weights into one directory:

  MODELS/MiniMax-Remover/                     github.com/zibojia/MiniMax-Remover
  MODELS/weights/minimax-remover/             huggingface zibojia/minimax-remover (vae, transformer, scheduler)
  MODELS/EffectErase/                         github.com/FudanCVL/EffectErase (then pip install -e it)
  MODELS/weights/Wan2.1-Fun-1.3B-InP/         huggingface alibaba-pai/Wan2.1-Fun-1.3B-InP
  MODELS/weights/effecterase/EffectErase.ckpt huggingface FudanCVL/EffectErase

Telea and SAM3 need nothing here (SAM3's weights download on first use; facebook/sam3 is gated: accept its
licence on the Hub and log in with `hf auth login` first).
"""

import subprocess
from pathlib import Path

REPOS = {
    'minimax': ('MiniMax-Remover', 'https://github.com/zibojia/MiniMax-Remover.git', '28e12b4'),
    'effecterase': ('EffectErase', 'https://github.com/FudanCVL/EffectErase.git', 'bcee0a5'),
}


def clone(models: Path, method: str):
    name, url, commit = REPOS[method]
    target = models / name
    if not target.exists():
        subprocess.run(['git', 'clone', url, str(target)], check=True)
    subprocess.run(['git', '-C', str(target), 'checkout', '-q', commit], check=True)


def fetch(method: str, models: Path):
    from huggingface_hub import hf_hub_download, snapshot_download

    models = Path(models)
    weights = models / 'weights'
    if method == 'minimax':
        clone(models, method)
        snapshot_download('zibojia/minimax-remover', local_dir=weights / 'minimax-remover',
                          allow_patterns=['vae/*', 'transformer/*', 'scheduler/*'])  # fmt: skip
    elif method == 'effecterase':
        clone(models, method)
        snapshot_download('alibaba-pai/Wan2.1-Fun-1.3B-InP', local_dir=weights / 'Wan2.1-Fun-1.3B-InP')
        hf_hub_download('FudanCVL/EffectErase', 'EffectErase.ckpt', local_dir=weights / 'effecterase')
        print(f'now install its package into the effecterase environment: pip install -e {models / "EffectErase"}')
    else:
        print(f'{method} needs nothing fetched')
        return
    print(f'{method} ready in {models}')
