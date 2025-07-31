import os
import csv
import json
from collections import defaultdict

def split_images_by_label_csv(
    csv_path: str,
    output_json: str,
    root_dir: str,                # 新增：所有相对路径的基准目录
    train_per_label: int = 45,
    val_per_label: int = 5,
    test_per_label: int = 5
):
    label_to_paths = defaultdict(list)
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel_path = row['image']
            # 把相对路径拼到 root_dir，再取绝对路径
            if rel_path.startswith("Img/"):
                rel_path = rel_path[4:]  # 去掉"Img/"
            abs_path = os.path.abspath(os.path.join(root_dir, rel_path))
            label_to_paths[row['label']].append(abs_path)

    train_list, val_list, test_list = [], [], []
    for label, paths in label_to_paths.items():
        paths = sorted(paths)
        if len(paths) < (train_per_label + val_per_label + test_per_label):
            raise ValueError(f"标签 {label} 数量不足")
        train_list.extend(paths[:train_per_label])
        val_list.extend(  paths[train_per_label:train_per_label + val_per_label])
        test_list.extend( paths[train_per_label + val_per_label:
                               train_per_label + val_per_label + test_per_label])

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"train": train_list, "val": val_list, "test": test_list},
                  f, ensure_ascii=False, indent=2)

    print(f"train: {len(train_list)}, val: {len(val_list)}, test: {len(test_list)}")

if __name__ == "__main__":
    split_images_by_label_csv(
    csv_path=r"D:\PythonProject1\english.csv",
    root_dir=r"D:\PythonProject1\Img",
    output_json=r"D:\PythonProject1\splits.json"
    )

