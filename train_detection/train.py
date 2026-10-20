"""Fine-tune a YOLOv8n detector with ultralytics and export it as a jaxtari .npz checkpoint.

Requires the train_detection extra: pip install -e ".[train_detection]" (ultralytics / torch).

    python train_detection/generate_dataset.py pong --size 96 96
    python train_detection/train.py pong --data train_detection/datasets/pong --out pong_gray96.npz

The start weights (--init) are an existing .npz checkpoint name or path, an ultralytics .pt file
(default: COCO-pretrained yolov8n.pt, downloaded by ultralytics if not present), or "scratch". For 1-channel datasets the first convolution of a 3-channel init is
summed over its input channels (exact for gray = R = G = B).
"""
import argparse
import copy
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def npz_state_dict(path):
    """jaxtari .npz checkpoint -> ultralytics/torch state_dict names (numpy arrays) and its spec."""
    with np.load(path, allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    spec = json.loads(str(arrays.pop("__spec__")))
    sd = {}
    for key, v in arrays.items():
        k = key
        k = re.sub(r"^(\d+)\.(box|cls)(\d+)\.2\.(weight|bias)$",
                   lambda m: f"{m[1]}.{'cv2' if m[2] == 'box' else 'cv3'}.{m[3]}.2.{m[4]}", k)  # plain head conv
        k = re.sub(r"^(\d+)\.(box|cls)(\d+)\.([01])\.", lambda m: f"{m[1]}.{'cv2' if m[2] == 'box' else 'cv3'}.{m[3]}.{m[4]}.", k)
        k = re.sub(r"\.m(\d+)\.", r".m.\1.", k)
        k = k.replace(".bn.mean", ".bn.running_mean").replace(".bn.var", ".bn.running_var")
        if not re.search(r"\.(cv2|cv3)\.\d+\.2\.(weight|bias)$", k):  # Conv modules: weight lives in .conv
            k = re.sub(r"\.weight$", ".conv.weight", k) if ".bn." not in k else k
        sd[f"model.{k}"] = v
    return sd, spec


def build_init(init, game, channels, nc, out):
    """Write an ultralytics start checkpoint (.pt) with `channels` input channels and `nc` classes."""
    import torch
    from ultralytics.nn.tasks import DetectionModel
    model = DetectionModel("yolov8n.yaml", ch=channels, nc=nc, verbose=False)
    if init != "scratch":
        if not str(init).endswith(".pt"):
            from jaxtari.detection import resolve_checkpoint
            src, _ = npz_state_dict(resolve_checkpoint(init))
            src = {k: torch.from_numpy(np.asarray(v, np.float32)) for k, v in src.items()}
        else:
            from ultralytics.utils.downloads import attempt_download_asset
            ck = torch.load(attempt_download_asset(str(init)), map_location="cpu", weights_only=False)
            src = (ck.get("ema") or ck["model"]).float().state_dict()
        sd = model.state_dict()
        loaded = 0
        for k, v in sd.items():
            if k not in src:
                continue
            w = src[k]
            if k == "model.0.conv.weight" and w.shape[1] != v.shape[1] and v.shape[1] == 1:
                w = w.sum(1, keepdim=True)  # RGB -> gray
            if w.shape == v.shape:
                sd[k], loaded = w, loaded + 1
        model.load_state_dict(sd)
        print(f"init: {loaded}/{len(sd)} tensors from {init}")
    from jaxtari.detection.labels import CLASS_NAMES
    model.names = CLASS_NAMES
    model.args = {"imgsz": 640}
    torch.save({"model": model.half(), "ema": None, "optimizer": None, "epoch": -1, "train_args": {}}, out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("game")
    ap.add_argument("--data", required=True, help="dataset dir from generate_dataset.py")
    ap.add_argument("--out", required=True, help="output .npz checkpoint")
    ap.add_argument("--init", default=None, help=".npz / .pt start weights or 'scratch'")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
    ap.add_argument("--conf", type=float, default=0.40)
    a = ap.parse_args()

    import yaml
    from ultralytics import YOLO
    from convert_ultralytics import convert

    data = Path(a.data)
    cfg = yaml.safe_load((data / "data.yaml").read_text())
    channels = int(cfg.get("channels", 3))
    img = next((data / "images" / "train").glob("*.png"))
    import cv2
    h, w = cv2.imread(str(img), cv2.IMREAD_UNCHANGED).shape[:2]
    rgb640 = channels == 3 and (w, h) == (640, 840)
    imgsz = 640 if rgb640 else max(w, h)
    if not rgb640 and (w % 32 or h % 32 or w != h):
        print(f"warning: {w}x{h} frames are letterboxed to {imgsz}; square multiples of 32 avoid any resizing")

    work = ROOT / "runs" / "train" / a.game
    work.mkdir(parents=True, exist_ok=True)
    init = a.init or "yolov8n.pt"
    start = build_init(init, a.game, channels, int(cfg["nc"]), work / "init.pt")

    t0 = time.time()
    YOLO(str(start)).train(
        data=str(data / "data.yaml"), imgsz=imgsz, epochs=a.epochs, batch=a.batch, device=a.device,
        project=str(work.parent), name=a.game, exist_ok=True, plots=False, verbose=False, seed=0, workers=2,
        fliplr=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0 if channels == 1 else 0.4,  # no mirroring / colour jitter
        scale=0.25, translate=0.1, patience=30, lr0=0.002, warmup_epochs=1)
    best = work / "weights" / "best.pt"
    res = YOLO(str(best)).val(data=str(data / "data.yaml"), imgsz=imgsz, device=a.device, verbose=False, plots=False)
    convert(best, a.out, a.game, "rgb" if rgb640 else "native", (640, 840) if rgb640 else (w, h), imgsz,
            conf=a.conf)
    print(json.dumps(dict(game=a.game, out=a.out, minutes=round((time.time() - t0) / 60, 1),
                          map50=round(float(res.box.map50), 4), map50_95=round(float(res.box.map), 4))))


if __name__ == "__main__":
    main()
