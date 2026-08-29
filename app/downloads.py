from pathlib import Path
from huggingface_hub import hf_hub_download
from .model_registry import path_for, VAE

def ensure(spec, progress=None):
    target = path_for(spec)
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    if progress: progress(0.03, desc=f"Downloading {spec.label} with Hugging Face Xet when available")
    downloaded = hf_hub_download(
        repo_id=spec.repo_id,
        filename=spec.filename,
        local_dir=str(target.parent),
    )
    return Path(downloaded)

def ensure_vae(progress=None):
    return ensure(VAE, progress)
