import time, random
from pathlib import Path
import gradio as gr
from .model_registry import BY_KEY, choice_list
from .seedvr import upscale as seedvr_upscale

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"outputs"; OUT.mkdir(exist_ok=True)

CSS = """
.gradio-container {max-width: 1780px !important; padding: 28px 34px 42px !important}
#app-title {text-align:center; margin:0; font-size:2.25rem !important; letter-spacing:-.02em}
#app-subtitle {text-align:center; opacity:.68; margin:6px 0 28px}
.panel {border:1px solid var(--border-color-primary); border-radius:18px; padding:22px !important; background:var(--background-fill-secondary)}
.section-title {margin:0 0 12px !important; opacity:.9}
#compare {min-height: 760px; border-radius:14px; overflow:hidden}
#compare img {object-fit:contain}
#status textarea {font-weight:600}
footer {display:none !important}
"""

VRAM_PRESETS = {
    # blocks, DiT offload, VAE offload, encode/decode tiling, tile sizes,
    # attention, cache DiT, cache VAE. High-VRAM profiles keep both resident.
    "Auto": (0, "none", "none", False, False, 1024, 128, 1024, 128, "sdpa", False, False),
    "4 GB / Emergency": (34, "cpu", "cpu", True, True, 512, 128, 512, 128, "sdpa", False, False),
    "6 GB / Very Low": (30, "cpu", "cpu", True, True, 512, 128, 512, 128, "sdpa", False, False),
    "8 GB / Low": (24, "cpu", "cpu", True, True, 768, 128, 768, 128, "sdpa", False, False),
    "10 GB / Compact": (18, "cpu", "cpu", True, True, 768, 128, 768, 128, "sdpa", False, False),
    "12 GB / Standard": (12, "cpu", "cpu", True, True, 1024, 128, 1024, 128, "sdpa", False, False),
    "16 GB / Comfortable": (6, "cpu", "cpu", False, True, 1024, 128, 1024, 128, "sdpa", False, True),
    "20 GB / High": (2, "none", "cpu", False, True, 1024, 128, 1024, 128, "sdpa", True, True),
    "24 GB / Quality": (0, "none", "none", False, False, 1024, 128, 1024, 128, "sdpa", True, True),
    "32 GB / Maximum": (0, "none", "none", False, False, 1536, 128, 1536, 128, "sdpa", True, True),
    "48 GB+ / Full Resident": (0, "none", "none", False, False, 2048, 128, 2048, 128, "sdpa", True, True),
    "Custom": None,
}

def profile_changed(profile):
    values = VRAM_PRESETS.get(profile)
    if values is None:
        return tuple(gr.update() for _ in range(12))
    return values

def process(image, model_key, resolution, max_resolution, seed, random_seed,
            pre_downscale, keep_alpha, memory_profile, cuda_device, attention_mode,
            dit_offload, vae_offload, blocks_to_swap, swap_io, non_blocking,
            enc_tiled, enc_tile_size, enc_tile_overlap, dec_tiled, dec_tile_size,
            dec_tile_overlap, cache_dit, cache_vae, debug,
            progress=gr.Progress()):
    if image is None:
        raise gr.Error("Please upload an image.")
    if random_seed:
        seed = random.randint(0, 2**31 - 1)

    spec = BY_KEY[model_key]
    progress(0.01, desc="Preparing image")

    if spec.family == "SeedVR":
        settings = {
            "resolution": resolution,
            "max_resolution": max_resolution,
            "seed": seed,
            "pre_downscale": pre_downscale,
            "keep_alpha": keep_alpha,
            "cuda_device": cuda_device,
            "attention_mode": attention_mode,
            "dit_offload": dit_offload,
            "vae_offload": vae_offload,
            "blocks_to_swap": blocks_to_swap,
            "swap_io": swap_io,
            "non_blocking": non_blocking,
            "enc_tiled": enc_tiled,
            "enc_tile_size": enc_tile_size,
            "enc_tile_overlap": enc_tile_overlap,
            "dec_tiled": dec_tiled,
            "dec_tile_size": dec_tile_size,
            "dec_tile_overlap": dec_tile_overlap,
            "cache_dit": cache_dit,
            "cache_vae": cache_vae,
            "debug": debug,
        }
        result = seedvr_upscale(image, spec, settings, progress)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = OUT / f"{stamp}_{model_key}.png"
    result.save(path)
    progress(1.0, desc="Done")
    return result, str(path), int(seed), f"Done — {result.width} × {result.height}"

with gr.Blocks(title="Seed-VR-2.5-Easy-GUI", fill_width=True) as demo:
    gr.Markdown("# ✦ SeedVR 2.5 · Easy GUI", elem_id="app-title")
    gr.Markdown(
        "**Local image upscaling workspace** · official FP16 · GGUF · CUDA · persistent model worker",
        elem_id="app-subtitle",
    )
    gr.Markdown(
        "[🌐 SeedVR 2.5 upstream](https://github.com/ByteDance-Seed/SeedVR) &nbsp;·&nbsp; "
        "[📖 Documentation](https://github.com/ByteDance-Seed/SeedVR#readme) &nbsp;·&nbsp; "
        "[📁 SeedVR model files](https://huggingface.co/MonsterMMORPG/Wan_GGUF/tree/main) &nbsp;·&nbsp; "
        "[🧠 GGUF model files](https://huggingface.co/cmeka/SeedVR2-GGUF/tree/main)",
        elem_id="project-links",
    )
    gr.Markdown(
        "> Image-only GUI for SeedVR 2.x/2.5 · models download on demand · "
        "SDPA with eager fallback · [project README](https://github.com/ByteDance-Seed/SeedVR#readme)",
        elem_id="project-scope",
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=430, elem_classes="panel"):
            gr.Markdown("### SeedVR model and memory", elem_classes="section-title")
            gr.Markdown("**Input image** · PNG, JPG or WebP")
            inp = gr.Image(label="Input Image", type="pil", image_mode="RGBA")
            model = gr.Dropdown(
                [(label, key) for label, key in choice_list() if BY_KEY[key].family == "SeedVR"],
                value="seed_7b_fp16", label="SeedVR Model")

            memory_profile = gr.Dropdown(list(VRAM_PRESETS.keys()), value="24 GB / Quality", label="VRAM Profile")
            gr.Markdown("Profiles configure memory-saving options and tiling automatically. Choose **Custom** for manual control.")
            with gr.Group():
                resolution = gr.Slider(256, 4096, 1080, step=8, label="Target Short Edge · output size")
                max_resolution = gr.Slider(0, 8192, 0, step=64, label="Maximum Long Edge · 0 = no limit")
                pre_downscale = gr.Dropdown([1,2,3], value=1, label="Pre-Downscale · saves VRAM")
                keep_alpha = gr.Checkbox(False, label="Preserve Alpha Channel · keep transparency")
                with gr.Row():
                    seed = gr.Number(value=42, precision=0, label="Seed · reproducible")
                    random_seed = gr.Checkbox(False, label="Random Seed · new each run")

            with gr.Accordion("Advanced SeedVR Settings", open=False):
                cuda_device = gr.Textbox(value="0", label="CUDA Device · GPU index")
                attention_mode = gr.Dropdown(["sdpa"], value="sdpa", label="Attention · SDPA", visible=False)
                dit_offload = gr.Dropdown(["none","cpu"], value="none", label="DiT Offload Device · saves VRAM")
                vae_offload = gr.Dropdown(["none","cpu"], value="none", label="VAE Offload Device · saves VRAM")
                blocks_to_swap = gr.Slider(0, 36, 0, step=1, label="Blocks To Swap · lower VRAM")
                swap_io = gr.Checkbox(False, label="Swap I/O Components · lower peak VRAM")
                non_blocking = gr.Checkbox(True, label="Use Non-Blocking Transfers · profile hint")
                enc_tiled = gr.Checkbox(False, label="Enable VAE Encode Tiling · large inputs")
                enc_tile_size = gr.Slider(256, 2048, 1024, step=64, label="Encode Tile Size · pixels")
                enc_tile_overlap = gr.Slider(0, 512, 128, step=16, label="Encode Tile Overlap · reduces seams")
                dec_tiled = gr.Checkbox(False, label="Enable VAE Decode Tiling · large outputs")
                dec_tile_size = gr.Slider(256, 2048, 1024, step=64, label="Decode Tile Size · pixels")
                dec_tile_overlap = gr.Slider(0, 512, 128, step=16, label="Decode Tile Overlap · reduces seams")
                cache_dit = gr.Checkbox(True, label="Keep DiT Loaded · faster repeats")
                cache_vae = gr.Checkbox(True, label="Keep VAE Loaded · faster repeats")
                debug = gr.Checkbox(False, label="Debug Logging · terminal diagnostics")

            run = gr.Button("Process Image", variant="primary", size="lg")
            status = gr.Textbox(label="Status", value="Ready", interactive=False)

        with gr.Column(scale=2, min_width=700, elem_classes="panel"):
            gr.Markdown("### Preview and result", elem_classes="section-title")
            compare = gr.ImageSlider(label="Before / After", elem_id="compare")
            with gr.Row():
                out = gr.Image(label="Processed Image", type="pil")
                download = gr.File(label="Download PNG")
            final_seed = gr.Number(label="Used Seed", interactive=False)

    memory_profile.change(
        profile_changed, memory_profile,
        [blocks_to_swap, dit_offload, vae_offload, enc_tiled, dec_tiled,
         enc_tile_size, enc_tile_overlap, dec_tile_size, dec_tile_overlap,
         attention_mode, cache_dit, cache_vae]
    )

    def run_and_compare(*args, **kwargs):
        result, path, used_seed, status = process(*args, **kwargs)
        original = args[0].convert("RGB")
        return (original, result.convert("RGB")), result, path, used_seed, status

    run.click(
        run_and_compare,
        [inp,model,resolution,max_resolution,seed,random_seed,pre_downscale,keep_alpha,
         memory_profile,cuda_device,attention_mode,dit_offload,vae_offload,blocks_to_swap,
         swap_io,non_blocking,enc_tiled,enc_tile_size,enc_tile_overlap,dec_tiled,
         dec_tile_size,dec_tile_overlap,cache_dit,cache_vae,debug],
        [compare,out,download,final_seed,status]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True, server_name="127.0.0.1", css=CSS)
