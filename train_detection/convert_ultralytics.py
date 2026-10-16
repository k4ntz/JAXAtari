"""Convert ultralytics YOLOv8 checkpoints (.pt) into torch-free jaxtari detector checkpoints (.npz).

This is the only place (besides training) that needs torch/ultralytics: ``pip install -e ".[train_detection]"``.

    # original 640px RGB checkpoints (frame = native RGB nearest-upscaled to 640x840)
    python train_detection/convert_ultralytics.py path/to/pong.pt pong_rgb640.npz --game pong \
        --input-mode rgb --frame-size 640 840 --imgsz 640
    # low-res checkpoints trained on JAXtari natively downscaled 96x96 grayscale frames
    python train_detection/convert_ultralytics.py best.pt pong_gray96.npz --game pong \
        --input-mode native --frame-size 96 96 --imgsz 96
"""
import argparse
import json
from pathlib import Path

import numpy as np

FORMAT = "jaxtari-yolov8"


def _conv(module, key, arrays, bn=True):
    """Store raw conv (+ BN) tensors under `key` and return its spec."""
    import torch.nn as nn
    conv = module.conv if bn else module
    arrays[f"{key}.weight"] = conv.weight.detach().cpu().numpy()
    spec = dict(key=key, k=conv.kernel_size[0], s=conv.stride[0], p=conv.padding[0], g=conv.groups, bn=bn)
    if bn:
        b = module.bn
        for name, t in (("weight", b.weight), ("bias", b.bias), ("mean", b.running_mean), ("var", b.running_var)):
            arrays[f"{key}.bn.{name}"] = t.detach().cpu().numpy()
        spec.update(eps=float(b.eps), act=not isinstance(module.act, nn.Identity))
    else:
        if conv.bias is not None:
            arrays[f"{key}.bias"] = conv.bias.detach().cpu().numpy()
        spec.update(act=False)
    return spec


def _layer(m, i, arrays):
    t = type(m).__name__
    spec = {"from": m.f}
    if t == "Conv":
        spec.update(type="Conv", conv=_conv(m, f"{i}", arrays))
    elif t == "C2f":
        spec.update(type="C2f", c=m.c, cv1=_conv(m.cv1, f"{i}.cv1", arrays), cv2=_conv(m.cv2, f"{i}.cv2", arrays),
                    m=[dict(cv1=_conv(b.cv1, f"{i}.m{j}.cv1", arrays), cv2=_conv(b.cv2, f"{i}.m{j}.cv2", arrays),
                            add=bool(b.add)) for j, b in enumerate(m.m)])
    elif t == "SPPF":
        k = m.m.kernel_size if isinstance(m.m.kernel_size, int) else m.m.kernel_size[0]
        spec.update(type="SPPF", k=k, n=getattr(m, "n", 3), add=bool(getattr(m, "add", False)),
                    cv1=_conv(m.cv1, f"{i}.cv1", arrays), cv2=_conv(m.cv2, f"{i}.cv2", arrays))
    elif t == "Upsample":
        assert m.mode == "nearest"
        spec.update(type="Upsample", scale=int(m.scale_factor))
    elif t == "Concat":
        spec.update(type="Concat")
    elif t == "Detect":
        # YOLOv8 head: cls branch = Conv, Conv, Conv2d (newer heads use DWConv), no one-to-one branch
        assert all(type(s[0]).__name__ == "Conv" for s in m.cv3) and not m.end2end, "only YOLOv8 heads supported"
        heads = []
        for h, (box, cls) in enumerate(zip(m.cv2, m.cv3)):
            heads.append({br: [_conv(seq[0], f"{i}.{br}{h}.0", arrays), _conv(seq[1], f"{i}.{br}{h}.1", arrays),
                               _conv(seq[2], f"{i}.{br}{h}.2", arrays, bn=False)]
                          for br, seq in (("box", box), ("cls", cls))})
        spec.update(type="Detect", nc=int(m.nc), reg_max=int(m.reg_max), strides=[float(s) for s in m.stride],
                    heads=heads)
    else:
        raise NotImplementedError(f"layer type {t} not supported")
    return spec


def convert(src, dst, game, input_mode, frame_size, imgsz, conf=0.40, iou=0.50):
    import torch
    from ultralytics import YOLO
    ck = torch.load(str(src), map_location="cpu", weights_only=False)
    half = next((ck.get("ema") or ck["model"]).parameters()).dtype == torch.float16  # as stored on disk
    model = YOLO(str(src)).model.float().eval()
    arrays, layers = {}, []
    for i, m in enumerate(model.model):
        layers.append(_layer(m, i, arrays))
    if half:  # ultralytics stores fp16 weights: keep that precision (lossless), halves the file size
        arrays = {k: v.astype(np.float16) for k, v in arrays.items()}
    spec = dict(format=FORMAT, version=1, game=game, names={int(k): v for k, v in model.names.items()},
                nc=int(model.model[-1].nc), save=sorted(int(s) for s in model.save), layers=layers,
                input_mode=input_mode, frame_size=list(frame_size), imgsz=imgsz, conf=conf, iou=iou,
                source=Path(src).name)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, __spec__=np.asarray(json.dumps(spec)), **arrays)
    return dst


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--game", required=True)
    ap.add_argument("--input-mode", choices=["rgb", "native"], required=True)
    ap.add_argument("--frame-size", type=int, nargs=2, required=True, metavar=("W", "H"))
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--iou", type=float, default=0.50)
    a = ap.parse_args()
    print(convert(a.src, a.dst, a.game, a.input_mode, a.frame_size, a.imgsz, a.conf, a.iou))
