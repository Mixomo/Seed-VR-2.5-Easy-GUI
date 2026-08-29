import json, os, sys, traceback, contextlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)

import inference_cli as cli

def make_args():
    sys.argv = ["inference_cli.py", "placeholder.png", "--output_format", "png"]
    return cli.parse_arguments()

args = make_args()
runner_cache = {}

for line in sys.stdin:
    try:
        request = json.loads(line)
        args.input = request["input"]
        args.output = request["output"]
        args.model_dir = request["model_dir"]
        args.dit_model = request["dit_model"]
        args.resolution = int(request["resolution"])
        args.max_resolution = int(request["max_resolution"])
        args.seed = int(request["seed"])
        args.cuda_device = str(request["cuda_device"])
        args.attention_mode = "sdpa"
        args.dit_offload_device = request["dit_offload"]
        args.vae_offload_device = request["vae_offload"]
        args.blocks_to_swap = int(request["blocks_to_swap"])
        args.swap_io_components = bool(request["swap_io"])
        args.vae_encode_tiled = bool(request["enc_tiled"])
        args.vae_encode_tile_size = int(request["enc_tile_size"])
        args.vae_encode_tile_overlap = int(request["enc_tile_overlap"])
        args.vae_decode_tiled = bool(request["dec_tiled"])
        args.vae_decode_tile_size = int(request["dec_tile_size"])
        args.vae_decode_tile_overlap = int(request["dec_tile_overlap"])
        args.cache_dit = bool(request["cache_dit"])
        args.cache_vae = bool(request["cache_vae"])
        args.chunk_size = 1 if (args.cache_dit or args.cache_vae) else 0
        cli.debug.enabled = bool(request.get("debug", False))
        cache = runner_cache if (args.cache_dit or args.cache_vae) else None
        with contextlib.redirect_stdout(sys.stderr):
            cli.process_single_file(args.input, args, [args.cuda_device], args.output,
                                    format_auto_detected=False, runner_cache=cache)
        print(json.dumps({"ok": True}), flush=True)
    except Exception as exc:
        traceback.print_exc()
        print(json.dumps({"ok": False, "error": str(exc)}), flush=True)
