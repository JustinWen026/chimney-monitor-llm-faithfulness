import glob
import os

# 你的第二份資料集（無火焰）絕對路徑
dataset_nofire_path = r"D:\NCKU_paper_Capstone_Project\chimney.v2-2026-08-14chimnetblack_smokywhite_smoky.yolov11"


def convert_label_id(split_name):
    # 搜尋 train/labels 或 valid/labels 下所有的 .txt 檔案
    txt_files = glob.glob(
        os.path.join(dataset_nofire_path, split_name, "labels", "*.txt")
    )
    modified_count = 0
    total_files = len(txt_files)

    for file_path in txt_files:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 0:
                # 原本第二份資料集中 2 代表 white_smoky，改為 3
                if parts[0] == "2":
                    parts[0] = "3"
                    modified_count += 1
                new_lines.append(" ".join(parts) + "\n")

        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

    print(
        f"[{split_name}] 掃描了 {total_files} 個標註檔，共將 {modified_count} 個 white_smoky 標籤由 2 修正為 3！"
    )


if __name__ == "__main__":
    print("開始修正標籤 ID...")
    convert_label_id("train")
    convert_label_id("valid")
    print("修正完成！")