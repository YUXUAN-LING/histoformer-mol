import os
import glob
import random
import re
# ============ 你要确认的路径 ============

TRAIN_LQ_DIR = "data/rain2/train/train/lq"   # *_rain.png
TRAIN_GT_DIR = "data/rain2/train/train/gt"     # *_clean.png

# VAL_LQ_DIR   = "data/rain2/test_a/test/data"
# VAL_GT_DIR   = "data/rain2/test_a/test/gt"

# TRAIN_LIST_PATH = "data/rain2/train_list_rain2.txt"
# VAL_LIST_PATH   = "data/rain2/val_list_rain2.txt"

######
# TRAIN_LQ_DIR = "data/haze/hazy"   # *_rain.png
# TRAIN_GT_DIR = "data/haze/clear"     # *_clean.png

VAL_LQ_DIR   = "data/rain2/test_a/val/lq"
VAL_GT_DIR   = "data/rain2/test_a/val/gt"

# TRAIN_LIST_PATH = "data/haze/train_list_haze.txt"
# VAL_LIST_PATH   = "data/haze/val_list_haze.txt"

TRAIN_LIST_PATH = "data/rain2/train_list_rain2.txt"
VAL_LIST_PATH   = "data/rain2/val_list_rain2.txt"

VAL_RATIO = 0.0   # 如果不需要验证集，就设为 0.0

# =======================================

# def collect_pairs(lq_dir, gt_dir):
#     """
#     根据编号前缀配对，例如：
#     GT: 0001.jpg
#     LQ: 0001_xxx.jpg, 0001_0.8_0.2.png 等
#     匹配规则：文件名前缀连续数字相同即可。
#     """
#     print(f"\n==> Collecting pairs from {lq_dir}")

#     # 匹配前缀数字的正则表达式
#     num_re = re.compile(r"^(\d+)")

#     # 找 LQ 和 GT 文件
#     lq_files = sorted(glob.glob(os.path.join(lq_dir, "*")))
#     gt_files = sorted(glob.glob(os.path.join(gt_dir, "*")))

#     # 建立 GT 的编号字典
#     gt_dict = {}
#     for gt in gt_files:
#         b = os.path.basename(gt)
#         m = num_re.match(b)
#         if m:
#             prefix = m.group(1)     # 提取数字编号
#             gt_dict[prefix] = os.path.abspath(gt)

#     print(f"GT 编号数: {len(gt_dict)}")

#     pairs = []
#     missing = []

#     # 遍历 LQ，按编号找对应 GT
#     for lq in lq_files:
#         b = os.path.basename(lq)
#         m = num_re.match(b)
#         if not m:
#             continue

#         prefix = m.group(1)

#         if prefix in gt_dict:
#             pairs.append((os.path.abspath(lq), gt_dict[prefix]))
#         else:
#             missing.append(lq)

#     print(f"成功配对: {len(pairs)}")
#     print(f"缺失 GT: {len(missing)}")
#     if missing:
#         print("前 10 个缺失 LQ 文件：")
#         for f in missing[:10]:
#             print(" -", f)

#     return pairs

def collect_pairs(lq_dir, gt_dir):
    """从目录中找 rain-clean 成对图像，按数字编号匹配"""
    print(f"\n==> Collecting pairs from {lq_dir}")

    lq_files = sorted(glob.glob(os.path.join(lq_dir, "*_rain.png")))
    print(f"找到 LQ (rain) 图像数量: {len(lq_files)}")

    pairs = []
    missing = []

    for lq in lq_files:
        name = os.path.basename(lq)         # eg. "000_rain.png"
        base = name.replace("_rain.png", "")  # "000"

        gt_name = f"{base}_clean.png"
        gt_path = os.path.join(gt_dir, gt_name)

        if os.path.exists(gt_path):
            pairs.append((os.path.abspath(lq), os.path.abspath(gt_path)))
        else:
            missing.append((lq, gt_path))

    print(f"成功配对: {len(pairs)}")
    print(f"缺失 GT: {len(missing)}")
    if missing:
        print("前 10 个缺失：")
        for a, b in missing[:10]:
            print(" LQ:", a)
            print(" GT:", b)

    return pairs


def split_train_val(pairs, ratio):
    """划分训练集和验证集"""
    if ratio <= 0:
        return pairs, []   # 不划分

    random.seed(42)
    random.shuffle(pairs)
    n_val = int(len(pairs) * ratio)
    return pairs[n_val:], pairs[:n_val]


def save_list(pairs, path):
    """保存列表文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for lq, gt in pairs:
            f.write(f"{lq} {gt}\n")
    print("写入:", path)


def main():

    # 训练集
    train_pairs = collect_pairs(TRAIN_LQ_DIR, TRAIN_GT_DIR)
    train_set, val_set = split_train_val(train_pairs, VAL_RATIO)

    save_list(train_set, TRAIN_LIST_PATH)
    save_list(val_set, VAL_LIST_PATH)

    # 验证集（独立目录）
    if VAL_RATIO == 0.0:
        print("\n==> 单独加载验证集目录...")
        val_pairs = collect_pairs(VAL_LQ_DIR, VAL_GT_DIR)
        save_list(val_pairs, VAL_LIST_PATH)


if __name__ == "__main__":
    main()
