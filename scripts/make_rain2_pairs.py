import os
import glob
import random

# ======== 需要你确认的路径 =========
# # LQ：有雨图
# LQ_DIR = "data/rain/train/lq"
# # GT：无雨图
# GT_DIR = "data/rain/train/gt"

# LQ：有雨图
# LQ_DIR = "data/rain2/train/train/data"
# # GT：无雨图
# GT_DIR = "data/rain2/train/train/gt"
# LQ：有雨图
LQ_DIR = "data/haze/hazy"
# GT：无雨图
GT_DIR = "data/haze/clear"

# TRAIN_LQ_DIR = "data/rain2/train/train/data"
# TRAIN_GT_DIR = "data/rain2/train/train/gt"

# VAL_LQ_DIR = "data/rain2/test_a/test/data"
# VAL_GT_DIR = "data/rain2/test_a/test/gt"

# TRAIN_LIST_PATH = "data/rain2/train_list_rain2.txt"
# VAL_LIST_PATH   = "data/rain2/val_list_rain2.txt"


# 输出的列表文件
# TRAIN_LIST_PATH = "data/rain2/train_list_rain2.txt"
# VAL_LIST_PATH   = "data/rain2/val_list_rain2.txt"
# 输出的列表文件
TRAIN_LIST_PATH = "data/haze/train_list_haze.txt"
VAL_LIST_PATH   = "data/haze/val_list_haze.txt"
VAL_RATIO = 0.2  # 20% 做验证集，你可以改成 0.1 等
# ==================================


def collect_pairs():
    lq_pattern = os.path.join(LQ_DIR, "*_rain_2_*.jpg")
    lq_files = sorted(glob.glob(lq_pattern))
    print("LQ 文件数量:", len(lq_files))

    pairs = []
    missing_gt = []

    for lq_path in lq_files:
        fname = os.path.basename(lq_path)
        # 例： 000000_00_rain_2_50.jpg -> base_id = 000000_00
        before, _ = fname.split("_rain_2_", 1)
        base_id = before  # 例如 "000000_00"
        gt_name = f"{base_id}_norain_2.png"   # 目标 GT 文件名
        gt_path = os.path.join(GT_DIR, gt_name)

        if os.path.exists(gt_path):
            pairs.append((os.path.abspath(lq_path), os.path.abspath(gt_path)))
        else:
            missing_gt.append((lq_path, gt_path))

    print("成功配对数量:", len(pairs))
    print("找不到 GT 的 LQ 数量:", len(missing_gt))
    if missing_gt:
        print("示例缺失配对（前 10 条）：")
        for lq, gt in missing_gt[:10]:
            print("  LQ:", lq)
            print("  期望 GT:", gt)

    return pairs


def split_train_val(pairs, val_ratio=0.2):
    random.seed(42)
    random.shuffle(pairs)
    n = len(pairs)
    n_val = int(n * val_ratio)
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]
    print(f"划分结果: train={len(train_pairs)}, val={len(val_pairs)}")
    return train_pairs, val_pairs


def save_list(pairs, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for lq, gt in pairs:
            f.write(f"{lq} {gt}\n")
    print("写入列表:", path)


def main():
    if not os.path.isdir(LQ_DIR):
        print("[ERROR] LQ_DIR 不存在:", LQ_DIR)
        return
    if not os.path.isdir(GT_DIR):
        print("[ERROR] GT_DIR 不存在:", GT_DIR)
        return

    pairs = collect_pairs()
    if not pairs:
        print("[ERROR] 没有成功配对的样本，请检查文件命名或路径。")
        return

    train_pairs, val_pairs = split_train_val(pairs, VAL_RATIO)
    save_list(train_pairs, TRAIN_LIST_PATH)
    save_list(val_pairs, VAL_LIST_PATH)


if __name__ == "__main__":
    main()
