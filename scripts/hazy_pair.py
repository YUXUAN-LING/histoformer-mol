import os
import glob
import random

# ====== 你需要改的路径 ======
# ROOT_DIR = "data/hazy/Test"     # Test 根目录
ROOT_DIR = "data/hazy/Train"     # Test 根目录
GT_DIR   = os.path.join(ROOT_DIR, "gt")
LQ_DIR   = os.path.join(ROOT_DIR, "hazy")   # 你的 hazy 图目录

OUT_LIST_PATH = "data/hazy/train_list_hazy.txt"
VAL_LIST_PATH = "data/hazy/val_list_hazy.txt"

MAX_PER_FOLDER = 10
VAL_RATIO = 0.0   # 不划分验证集就设 0
SEED = 42
# ===========================

IMG_EXTS = (".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff")


def natural_key(x):
    """让 00001.jpg, 00002.jpg 这种按数字顺序排"""
    base = os.path.splitext(os.path.basename(x))[0]
    try:
        return int(base)
    except:
        return base


def collect_pairs():
    pairs = []
    missing = []

    subfolders = sorted([d for d in os.listdir(GT_DIR)
                         if os.path.isdir(os.path.join(GT_DIR, d))])

    print("GT 子文件夹数量:", len(subfolders))

    for sub in subfolders:
        gt_sub = os.path.join(GT_DIR, sub)
        lq_sub = os.path.join(LQ_DIR, sub)

        if not os.path.isdir(lq_sub):
            print(f"[WARN] hazy 中缺少子文件夹: {sub}")
            continue

        # 找 gt 里所有图片
        gt_imgs = sorted(
            [p for p in glob.glob(os.path.join(gt_sub, "*"))
             if os.path.splitext(p)[1].lower() in IMG_EXTS],
            key=natural_key
        )

        # 只取前三张
        gt_imgs = gt_imgs[:MAX_PER_FOLDER]

        for gt_path in gt_imgs:
            name = os.path.basename(gt_path)
            lq_path = os.path.join(lq_sub, name)

            if os.path.exists(lq_path):
                pairs.append((os.path.abspath(lq_path), os.path.abspath(gt_path)))
            else:
                missing.append((gt_path, lq_path))

        print(f"{sub}: 取到 {len(gt_imgs)} 张, 成功配对 {len(gt_imgs)-len([m for m in missing if sub in m[0]])}")

    print("\n总配对数:", len(pairs))
    print("缺失配对数:", len(missing))
    if missing:
        print("示例缺失（前10个）:")
        for gt, lq in missing[:10]:
            print(" GT:", gt)
            print(" LQ:", lq)

    return pairs


def split_train_val(pairs, val_ratio):
    if val_ratio <= 0:
        return pairs, []
    random.seed(SEED)
    random.shuffle(pairs)
    n_val = int(len(pairs) * val_ratio)
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]
    print(f"划分: train={len(train_pairs)} val={len(val_pairs)}")
    return train_pairs, val_pairs


def save_list(pairs, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for lq, gt in pairs:
            f.write(f"{lq} {gt}\n")
    print("写入列表:", path)


def main():
    assert os.path.isdir(GT_DIR), f"GT_DIR 不存在: {GT_DIR}"
    assert os.path.isdir(LQ_DIR), f"LQ_DIR 不存在: {LQ_DIR}"

    pairs = collect_pairs()
    if not pairs:
        print("[ERROR] 没有配到任何样本，请检查路径或命名")
        return

    train_pairs, val_pairs = split_train_val(pairs, VAL_RATIO)
    save_list(train_pairs, OUT_LIST_PATH)
    save_list(val_pairs, VAL_LIST_PATH)


if __name__ == "__main__":
    main()
