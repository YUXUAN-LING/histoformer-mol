import os
import glob

# =============== 你只需要改这里 ===================
LQ_DIR = "data/rain/val/lq"
GT_DIR = "data/rain/val/gt"
OUT_LIST = "data/rain/val_list_rain2.txt"
# ===================================================


def main():
    if not os.path.isdir(LQ_DIR):
        print("[ERROR] LQ_DIR 不存在:", LQ_DIR)
        return
    if not os.path.isdir(GT_DIR):
        print("[ERROR] GT_DIR 不存在:", GT_DIR)
        return

    # 匹配所有 _rain_2 的 LQ 图像
    lq_files = sorted(glob.glob(os.path.join(LQ_DIR, "*_rain_2_*.jpg")))
    print("找到 LQ 验证图像数量:", len(lq_files))

    pairs = []
    missing = []

    for lq in lq_files:
        fname = os.path.basename(lq)

        # 例如 000000_00_rain_2_50.jpg → base_id=000000_00
        before, _ = fname.split("_rain_2_", 1)
        base_id = before
        gt_name = f"{base_id}_norain_2.png"
        gt_path = os.path.join(GT_DIR, gt_name)

        if os.path.exists(gt_path):
            pairs.append((os.path.abspath(lq), os.path.abspath(gt_path)))
        else:
            missing.append((lq, gt_path))

    print("成功配对数量:", len(pairs))
    print("找不到 GT 的数量:", len(missing))
    if missing:
        print("示例缺失配对（前 10 个）：")
        for a,b in missing[:10]:
            print("  LQ:", a)
            print("  期望 GT:", b)

    # 写入文本
    os.makedirs(os.path.dirname(OUT_LIST), exist_ok=True)
    with open(OUT_LIST, "w", encoding="utf-8") as f:
        for lq, gt in pairs:
            f.write(f"{lq} {gt}\n")

    print("已写入验证集列表:", OUT_LIST)


if __name__ == "__main__":
    main()
