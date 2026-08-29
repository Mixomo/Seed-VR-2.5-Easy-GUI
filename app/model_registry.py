from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    family: str
    repo_id: str
    filename: str
    scale: int = 0
    needs_vae: bool = False

GGUF_REPO = "cmeka/SeedVR2-GGUF"
SEED_REPO = "MonsterMMORPG/Wan_GGUF"

SPECS = [
    ModelSpec("seed_3b_fp16","SeedVR 3B FP16","SeedVR",SEED_REPO,"seedvr2_ema_3b_fp16.safetensors",needs_vae=True),
    ModelSpec("seed_7b_fp16","SeedVR 7B FP16","SeedVR",SEED_REPO,"seedvr2_ema_7b_fp16.safetensors",needs_vae=True),
    ModelSpec("seed_7b_sharp_fp16","SeedVR 7B Sharp FP16","SeedVR",SEED_REPO,"seedvr2_ema_7b_sharp_fp16.safetensors",needs_vae=True),
    ModelSpec("seed_3b_fp8","SeedVR 3B FP8","SeedVR",SEED_REPO,"seedvr2_ema_3b_fp8_e4m3fn.safetensors",needs_vae=True),
    ModelSpec("seed_7b_fp8","SeedVR 7B FP8 Mixed Block35","SeedVR",SEED_REPO,"seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors",needs_vae=True),
    ModelSpec("seed_7b_sharp_fp8","SeedVR 7B Sharp FP8 Mixed Block35","SeedVR",SEED_REPO,"seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors",needs_vae=True),
    ModelSpec("seed_3b_q8","SeedVR 3B GGUF Q8_0","SeedVR",GGUF_REPO,"seedvr2_ema_3b-Q8_0.gguf",needs_vae=True),
    ModelSpec("seed_7b_q8","SeedVR 7B GGUF Q8_0","SeedVR",GGUF_REPO,"seedvr2_ema_7b-Q8_0.gguf",needs_vae=True),
    ModelSpec("seed_7b_sharp_q8","SeedVR 7B Sharp GGUF Q8_0","SeedVR",GGUF_REPO,"seedvr2_ema_7b_sharp-Q8_0.gguf",needs_vae=True),
    ModelSpec("seed_3b_q4","SeedVR 3B GGUF Q4_K_M","SeedVR",GGUF_REPO,"seedvr2_ema_3b-Q4_K_M.gguf",needs_vae=True),
    ModelSpec("seed_7b_q4","SeedVR 7B GGUF Q4_K_M","SeedVR",GGUF_REPO,"seedvr2_ema_7b-Q4_K_M.gguf",needs_vae=True),
    ModelSpec("seed_7b_sharp_q4","SeedVR 7B Sharp GGUF Q4_K_M","SeedVR",GGUF_REPO,"seedvr2_ema_7b_sharp-Q4_K_M.gguf",needs_vae=True),
]
VAE = ModelSpec("seed_vae","SeedVR VAE FP16","SeedVR",SEED_REPO,"ema_vae_fp16.safetensors")
BY_KEY = {s.key:s for s in SPECS}
def path_for(spec): return MODELS / spec.family.lower() / spec.filename
def choice_list(): return [(s.label,s.key) for s in SPECS]
