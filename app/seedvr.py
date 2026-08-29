import subprocess, sys, shutil, os, json, threading
from pathlib import Path
from PIL import Image
from .downloads import ensure, ensure_vae

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
TEMP = ROOT / "temp" / "seedvr"
_WORKER = None
_WORKER_LOCK = threading.Lock()

def _add(cmd, name, value):
    if value is None: return
    cmd.extend([name, str(value)])

def upscale(image, spec, settings, progress=None):
    global _WORKER
    model_path = ensure(spec, progress)
    ensure_vae(progress)
    cli = BACKEND / "inference_cli.py"
    if not cli.exists():
        raise RuntimeError("SeedVR backend is missing. Run install.bat again.")

    TEMP.mkdir(parents=True, exist_ok=True)
    in_path = TEMP / "input.png"
    out_dir = TEMP / "output"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    work = image.convert("RGBA" if settings["keep_alpha"] else "RGB")
    if settings["pre_downscale"] > 1:
        work.thumbnail((work.width // settings["pre_downscale"], work.height // settings["pre_downscale"]), Image.Resampling.LANCZOS)
    work.save(in_path)

    request = {
        "input": str(in_path), "output": str(out_dir), "model_dir": str(model_path.parent),
        "dit_model": spec.filename, "resolution": int(settings["resolution"]),
        "max_resolution": int(settings["max_resolution"]), "seed": int(settings["seed"]),
        "cuda_device": str(settings["cuda_device"]), "dit_offload": settings["dit_offload"],
        "vae_offload": settings["vae_offload"], "blocks_to_swap": int(settings["blocks_to_swap"]),
        "swap_io": settings["swap_io"], "enc_tiled": settings["enc_tiled"],
        "enc_tile_size": int(settings["enc_tile_size"]), "enc_tile_overlap": int(settings["enc_tile_overlap"]),
        "dec_tiled": settings["dec_tiled"], "dec_tile_size": int(settings["dec_tile_size"]),
        "dec_tile_overlap": int(settings["dec_tile_overlap"]), "cache_dit": settings["cache_dit"],
        "cache_vae": settings["cache_vae"], "debug": settings["debug"]}
    with _WORKER_LOCK:
        if _WORKER is None or _WORKER.poll() is not None:
            _WORKER = subprocess.Popen([sys.executable, str(ROOT / "app" / "seedvr_worker.py")],
                cwd=str(BACKEND), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=None, text=True, bufsize=1)
        _WORKER.stdin.write(json.dumps(request) + "\n"); _WORKER.stdin.flush()
        response = None
        while response is None:
            line = _WORKER.stdout.readline()
            if not line:
                code = _WORKER.poll()
                raise RuntimeError(f"SeedVR worker stopped before replying (exit code: {code}). Check the console for its traceback.")
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                print(f"[SeedVR worker] {line.rstrip()}", flush=True)
                continue
            if isinstance(candidate, dict) and "ok" in candidate:
                response = candidate
    if not response.get("ok"):
        raise RuntimeError(response.get("error", "SeedVR worker failed."))
    candidates = sorted(list(out_dir.rglob("*.png")) + list(out_dir.rglob("*.jpg")) + list(out_dir.rglob("*.webp")))
    if not candidates: raise RuntimeError("SeedVR finished without producing an image.")
    return Image.open(candidates[0]).convert("RGBA" if settings["keep_alpha"] else "RGB")

    cmd = [
        sys.executable, str(cli), str(in_path),
        "--output", str(out_dir),
        "--model_dir", str(model_path.parent),
        "--dit_model", spec.filename,
        "--resolution", str(int(settings["resolution"])),
        "--max_resolution", str(int(settings["max_resolution"])),
        "--seed", str(int(settings["seed"])),
        "--batch_size", "1",
        "--cuda_device", str(settings["cuda_device"]),
        "--attention_mode", "sdpa",
        "--dit_offload_device", settings["dit_offload"],
        "--vae_offload_device", settings["vae_offload"],
        "--blocks_to_swap", str(int(settings["blocks_to_swap"])),
        "--vae_encode_tile_size", str(int(settings["enc_tile_size"])),
        "--vae_encode_tile_overlap", str(int(settings["enc_tile_overlap"])),
        "--vae_decode_tile_size", str(int(settings["dec_tile_size"])),
        "--vae_decode_tile_overlap", str(int(settings["dec_tile_overlap"])),
    ]
    if settings["swap_io"]: cmd.append("--swap_io_components")
    # The backend CLI has no --use_non_blocking option.
    if settings["enc_tiled"]: cmd.append("--vae_encode_tiled")
    if settings["dec_tiled"]: cmd.append("--vae_decode_tiled")
    if settings["cache_dit"]: cmd.append("--cache_dit")
    if settings["cache_vae"]: cmd.append("--cache_vae")
    if settings["debug"]: cmd.append("--debug")
    # The backend keeps its model cache only in streaming mode. A GUI run is
    # one image, so use one-frame streaming when either residency checkbox is
    # enabled; otherwise the CLI deliberately frees the model at the end.
    if settings["cache_dit"] or settings["cache_vae"]:
        cmd.extend(["--chunk_size", "1"])

    print(
        "[SeedVR] requested attention=%s | cache_dit=%s | cache_vae=%s | chunk_size=%s"
        % (settings["attention_mode"], settings["cache_dit"], settings["cache_vae"],
           "1" if (settings["cache_dit"] or settings["cache_vae"]) else "default"),
        flush=True,
    )

    if progress: progress(0.50, desc="Running SeedVR 2.5")
    proc = subprocess.run(cmd, cwd=str(BACKEND), text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout, end="", flush=True)
    if proc.stderr:
        print(proc.stderr, end="", flush=True)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "SeedVR failed.")[-5000:])
    candidates = sorted(list(out_dir.rglob("*.png")) + list(out_dir.rglob("*.jpg")) + list(out_dir.rglob("*.webp")))
    if not candidates:
        raise RuntimeError("SeedVR finished without producing an image.")
    return Image.open(candidates[0]).convert("RGBA" if settings["keep_alpha"] else "RGB")
