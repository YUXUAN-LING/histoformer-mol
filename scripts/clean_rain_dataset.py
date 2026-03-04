import os
import re
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=str, required=True,
        help="dataset root folder, e.g., data/image_2_3_rain50"
    )
    parser.add_argument(
        "--mode", type=str, default="dry",
        choices=["dry", "delete"],
        help="dry: only show files to delete; delete: actually remove them"
    )
    args = parser.parse_args()

    root = args.root
    mode = args.mode

    print("========== clean_rain_dataset ==========")
    print("root:", root)
    print("mode:", mode)

    # 路径检查
    if not os.path.isdir(root):
        print(f"[ERROR] 目录不存在: {root}")
        return

    all_files = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]
    print("目录下文件总数:", len(all_files))

    # 保留的模式：000000_00_rain_2_50.jpg 这种
    # keep_pattern = re.compile(r"^\d{6}_00_rain_2_.*\.jpg$", re.IGNORECASE)
    keep_pattern = re.compile(r"^\d{6}_00_norain_2\.png$", re.IGNORECASE)


    keep_list = []
    del_list = []

    for fname in all_files:
        fpath = os.path.join(root, fname)
        if keep_pattern.match(fname):
            keep_list.append(fpath)
        else:
            del_list.append(fpath)

    print("-----------------------------------------")
    print("符合保留规则的文件数量:", len(keep_list))
    print("将被删除的文件数量:", len(del_list))
    print("-----------------------------------------")

    # 预览前 50 个要删的
    print("示例将被删除的文件（最多列出 50 个）:")
    for f in del_list[:50]:
        print("  [DEL]", f)
    if len(del_list) > 50:
        print("  ... (还有更多未显示)")

    if mode == "dry":
        print("\n>>> 当前为 dry run（预览模式），没有删除任何文件。")
        print(">>> 如果确认无误，请再次运行并加上: --mode delete")
        return

    # 真正删除
    print("\n>>> 开始删除文件 ...")
    cnt_ok = 0
    cnt_err = 0
    for f in del_list:
        try:
            os.remove(f)
            cnt_ok += 1
        except Exception as e:
            print("[ERROR] 删除失败:", f, "原因:", e)
            cnt_err += 1

    print(">>> 删除完成，成功删除:", cnt_ok, "个文件，失败:", cnt_err, "个。")

if __name__ == "__main__":
    main()
