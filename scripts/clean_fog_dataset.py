import os
import argparse
import re

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="包含所有图片的文件夹")
    ap.add_argument("--ext", default=".jpg", help="图片后缀（默认 .jpg）")
    ap.add_argument("--mode", choices=["dry", "clean"], default="dry",
                    help="dry = 只预览要删除的文件；clean = 真正删除")
    args = ap.parse_args()

    root = args.root
    ext = args.ext.lower()

    if not os.path.isdir(root):
        print("[ERROR] 路径不存在:", root)
        return
    ###############################
    # 6位数字 + .jpg
    pattern = re.compile(r"^(\d{5})" + re.escape(ext) + r"$", re.IGNORECASE)
    ################################
    files = [f for f in os.listdir(root) if f.lower().endswith(ext)]
    files.sort()

    to_delete = []
    to_keep = []

    for fname in files:
        m = pattern.match(fname)
        if not m:
            # 命名不符合直接删除
            to_delete.append(fname)
            continue

        seq = int(m.group(1))  # 获取序号
        if seq % 30 == 0:
            to_keep.append(fname)      # 保留
        else:
            to_delete.append(fname)    # 删除

    print(f"[INFO] 总文件数: {len(files)}")
    print(f"[INFO] 保留的文件（总 {len(to_keep)} 个）: 前 10 个:")
    for x in to_keep[:10]:
        print("   ", x)

    print(f"[INFO] 将要删除的文件（总 {len(to_delete)} 个）: 前 10 个:")
    for x in to_delete[:10]:
        print("   ", x)

    if args.mode == "dry":
        print("\n[DRY RUN] 只显示结果，不会删除任何文件。")
        return

    # 真正删除
    print("\n[CLEAN] 正在删除文件...")
    removed = 0
    for fname in to_delete:
        path = os.path.join(root, fname)
        try:
            os.remove(path)
            removed += 1
        except Exception as e:
            print("[ERROR] 无法删除:", path, "|", e)

    print(f"[DONE] 已删除 {removed} 个文件。")


if __name__ == "__main__":
    main()
