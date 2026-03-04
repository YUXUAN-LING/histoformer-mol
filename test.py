# test.py
import os
import torch
from torch import amp
import torch.nn.functional as F
from lora_adapters.utils import build_histoformer, load_image, save_image

@torch.no_grad()
def run_once(model, x, device="cuda"):
    """整图推理（只 pad 到 8 的倍数）"""
    B, C, H, W = x.shape
    new_H = (H + 7) // 8 * 8
    new_W = (W + 7) // 8 * 8
    pad_bottom = new_H - H
    pad_right = new_W - W

    if pad_bottom > 0 or pad_right > 0:
        x = F.pad(x, (0, pad_right, 0, pad_bottom), mode="reflect")

    with amp.autocast(device_type="cuda", enabled=(device == "cuda")):
        out = model(x)

    return out[:, :, :H, :W]


@torch.no_grad()
def tile_inference(model, x, tile=640, overlap=32, device="cuda"):
    """
    分块推理避免 OOM（与你 infer_folder_lora_compare.py 一致）:contentReference[oaicite:2]{index=2}
    """
    B, C, H, W = x.shape
    if H <= tile and W <= tile:
        return run_once(model, x, device)

    stride = tile - overlap
    ys = list(range(0, H, stride))
    xs = list(range(0, W, stride))
    if ys[-1] + tile < H: ys.append(max(H - tile, 0))
    if xs[-1] + tile < W: xs.append(max(W - tile, 0))
    ys = sorted(set(ys)); xs = sorted(set(xs))

    out_full = torch.zeros((1, C, H, W), device=x.device, dtype=torch.float32)
    weight   = torch.zeros((1, 1, H, W), device=x.device, dtype=torch.float32)

    for y0 in ys:
        for x0 in xs:
            y1 = min(y0 + tile, H)
            x1 = min(x0 + tile, W)
            patch = x[:, :, y0:y1, x0:x1]
            ph, pw = patch.shape[2:]

            # pad patch 到 8 的倍数
            new_ph = (ph + 7) // 8 * 8
            new_pw = (pw + 7) // 8 * 8
            pad_bottom = new_ph - ph
            pad_right  = new_pw - pw
            if pad_bottom > 0 or pad_right > 0:
                patch = F.pad(patch, (0, pad_right, 0, pad_bottom), mode="reflect")

            with amp.autocast(device_type="cuda", enabled=(device == "cuda")):
                out_patch = model(patch)

            out_patch = out_patch[:, :, :ph, :pw]
            out_full[:, :, y0:y1, x0:x1] += out_patch
            weight[:, :, y0:y1, x0:x1]   += 1.0

    out_full = out_full / torch.clamp(weight, min=1.0)
    return out_full


def main():
    base_ckpt = "pretrained_models/histoformer_base.pth"
    yaml_file = "lora_adapters/configs/histoformer_mol.yaml"
    input_img = "data/hazy/Test/hazy/C005/00001.JPG"      # 改成你的图
    out_img   = "results/test_out.png"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    net = build_histoformer(weights=base_ckpt, yaml_file=yaml_file)
    net.to(device).eval()

    x, _ = load_image(input_img)  # [1,3,H,W]
    x = x.to(device)

    # ✅ 自动 tile 推理
    out = tile_inference(net, x, tile=640, overlap=32, device=device)

    save_image(out, out_img)
    print("saved to", out_img)


if __name__ == "__main__":
    main()
