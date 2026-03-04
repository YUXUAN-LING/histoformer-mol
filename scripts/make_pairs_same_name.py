import os

# =============== 你只需要确认这几个路径 =================

# TRAIN_LQ_DIR = "data/snow/train/lq"
# TRAIN_GT_DIR = "data/snow/train/gt"

# VAL_LQ_DIR = "data/snow/val/lq"
# VAL_GT_DIR = "data/snow/val/gt"

# TRAIN_LIST_PATH = "data/snow/train_list_snow.txt"
# VAL_LIST_PATH   = "data/snow/val_list_snow.txt"

################

TRAIN_LQ_DIR = "data/blur/train/lq"
TRAIN_GT_DIR = "data/blur/train/gt"

VAL_LQ_DIR = "data/blur/val/lq"
VAL_GT_DIR = "data/blur/val/gt"

TRAIN_LIST_PATH = "data/blur/train_list_blur.txt"
VAL_LIST_PATH   = "data/blur/val_list_blur.txt"

#################
# TRAIN_LQ_DIR = "data/haze/train/lq"
# TRAIN_GT_DIR = "data/haze/train/gt"

# VAL_LQ_DIR = "data/haze/val/lq"
# VAL_GT_DIR = "data/haze/val/gt"

# TRAIN_LIST_PATH = "data/haze/train_list_haze.txt"
# VAL_LIST_PATH   = "data/haze/val_list_haze.txt"
############
TRAIN_LQ_DIR = "data/rainy/train/lq"
TRAIN_GT_DIR = "data/rainy/train/gt"

VAL_LQ_DIR = "data/rainy/val/lq"
VAL_GT_DIR = "data/rainy/val/gt"

TRAIN_LIST_PATH = "data/rainy/train_list_rainy.txt"
VAL_LIST_PATH   = "data/rainy/val_list_rainy.txt"
#############
TRAIN_LQ_DIR = "data/CDD-11_train/snow"
TRAIN_GT_DIR = "data/CDD-11_train/clear"

VAL_LQ_DIR = "samples/snow"
VAL_GT_DIR = "samples/clear"

TRAIN_LIST_PATH = "data/CDD-11_train/train_list_snow1.txt"
VAL_LIST_PATH   = "data/CDD-11_train/val_list_snow1.txt"

###############################################
TRAIN_LQ_DIR = "data/CDD-11_train/haze_snow"
TRAIN_GT_DIR = "data/CDD-11_train/clear"

VAL_LQ_DIR = "test_data/Rain100L/input"
VAL_GT_DIR = "test_data/Rain100L/target"

TRAIN_LIST_PATH = "data/txt_lists/train_list_haze_snow.txt"
VAL_LIST_PATH   = "test_data/Rain100L/val_list_Rain100L.txt"




###############################################
# 支持的图片后缀
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# =======================================================


def build_pairs_same_name(lq_dir, gt_dir):
    if not os.path.isdir(lq_dir):
        print("[ERROR] LQ_DIR 不存在:", lq_dir)
        return []

    if not os.path.isdir(gt_dir):
        print("[ERROR] GT_DIR 不存在:", gt_dir)
        return []

    files = []
    for name in os.listdir(lq_dir):
        ext = os.path.splitext(name)[1].lower()
        if ext in IMG_EXTS:
            files.append(name)
    files.sort()

    print(f"[INFO] 在 {lq_dir} 找到 LQ 图像数量: {len(files)}")

    pairs = []
    missing = []

    for name in files:
        lq_path = os.path.join(lq_dir, name)
        gt_path = os.path.join(gt_dir, name)
        if os.path.exists(gt_path):
            pairs.append((os.path.abspath(lq_path), os.path.abspath(gt_path)))
        else:
            missing.append((lq_path, gt_path))

    print(f"[INFO] 成功配对数量: {len(pairs)}")
    print(f"[INFO] 找不到 GT 的数量: {len(missing)}")
    if missing:
        print("[WARN] 示例缺失（前 10 条）：")
        for lq, gt in missing[:10]:
            print("  LQ:", lq)
            print("  期望 GT:", gt)

    return pairs


def save_list(pairs, path):
    if not pairs:
        print("[WARN] 没有样本，跳过写入:", path)
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for lq, gt in pairs:
            f.write(f"{lq} {gt}\n")
    print("[INFO] 写入列表:", path)


def main():
    print("========== 构建 xxx训练集配对 ==========")
    train_pairs = build_pairs_same_name(TRAIN_LQ_DIR, TRAIN_GT_DIR)
    save_list(train_pairs, TRAIN_LIST_PATH)

    print("========== 构建 xxx 验证集配对 ==========")
    val_pairs = build_pairs_same_name(VAL_LQ_DIR, VAL_GT_DIR)
    save_list(val_pairs, VAL_LIST_PATH)


if __name__ == "__main__":
    main()
