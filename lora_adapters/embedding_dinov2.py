# lora_adapters/embedding_dinov2.py
from pathlib import Path
import glob, numpy as np, torch, timm
from PIL import Image
import torchvision.transforms as T

class DINOv2Embedder:
    def __init__(
        self,
        device=None,
        size=518,
        ckpt_path: str = "weights/dinov2/vit_base_patch14_dinov2.lvd142m-384.pth"
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        ckpt = Path(ckpt_path)
        if ckpt.is_file():
            print(f"[DINOv2] 使用本地权重: {ckpt}")
            # 和 build_prototypes.py 一致：本地加载
            self.model = timm.create_model(
                'vit_base_patch14_dinov2.lvd142m',
                pretrained=False
            ).to(self.device).eval()

            state = torch.load(ckpt, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.model.load_state_dict(state, strict=False)
        else:
            # fallback：找不到本地文件才尝试 pretrained=True（需要联网）
            print(f"[DINOv2] 本地 ckpt 未找到: {ckpt}，尝试 pretrained=True（需要联网）")
            self.model = timm.create_model(
                'vit_base_patch14_dinov2.lvd142m',
                pretrained=True
            ).to(self.device).eval()

        self.tfm = T.Compose([
            T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
        ])

    def embed_image(self, img_or_path) -> np.ndarray:
        if isinstance(img_or_path, (str, Path)):
            img = Image.open(img_or_path).convert('RGB')
        else:
            img = img_or_path
        x = self.tfm(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model.forward_features(x)
            if isinstance(out, dict) and "x_norm_clstoken" in out:
                v = out["x_norm_clstoken"]          # [1, C]
            elif isinstance(out, torch.Tensor) and out.dim() == 3:
                v = out[:, 0]                        # [1, C]
            else:
                v = out.mean(dim=1)
            v = torch.nn.functional.normalize(v, dim=-1)
        return v.squeeze(0).cpu().numpy()

    # 下面 build_average / save_average 保持不变
    def build_average(self, train_dir: Path, max_images=300, exts=(".png",".jpg",".jpeg",".bmp",".tif")) -> np.ndarray:
        files = []
        for e in exts: files += glob.glob(str(Path(train_dir)/f"**/*{e}"), recursive=True)
        if not files: raise FileNotFoundError(f"No images under {train_dir}")
        files = files[:max_images]
        embs = [self.embed_image(p) for p in files]
        return np.stack(embs,0).mean(0)

    def save_average(self, out_path: Path, train_dir: Path, **kw):
        avg = self.build_average(train_dir, **kw)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, avg)
        return avg
