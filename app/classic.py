import torch, numpy as np
from PIL import Image
from spandrel import ModelLoader
from .downloads import ensure

def upscale(image, spec, tile=0, progress=None):
    path = ensure(spec, progress)
    if progress: progress(0.5, desc="Loading image model")
    model = ModelLoader().load_from_file(str(path)).eval()
    if torch.cuda.is_available():
        model.cuda()
    arr = np.array(image.convert("RGB")).astype("float32") / 255.0
    x = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)
    if torch.cuda.is_available(): x=x.cuda()
    with torch.inference_mode():
        y = model(x)
    y = y.squeeze(0).permute(1,2,0).clamp(0,1).float().cpu().numpy()
    return Image.fromarray((y*255+0.5).astype("uint8"))
