import os
import json
import shutil
from tqdm import tqdm

SRC_ROOT = "data/DocLayNet"

FOLDERS = ["train_data_docling", "val_data_docling", "test_data_docling"]

def transform_graph(graph):
    # 1. odstráň YOLO veci
    graph.pop("yolo_nodes", None)

    # 2. sem budeš dopĺňať DOCLING výstup
    graph["docling_nodes"] = []

    return graph


for folder in FOLDERS:
    src_folder = os.path.join(SRC_ROOT, folder)

    json_files = [f for f in os.listdir(src_folder) if f.endswith(".json")]

    for f in tqdm(json_files, desc=folder):

        src_path = os.path.join(src_folder, f)

        with open(src_path, "r", encoding="utf-8") as fin:
            graph = json.load(fin)

        # transformácia
        graph = transform_graph(graph)

        with open(src_path, "w", encoding="utf-8") as fout:
            json.dump(graph, fout, ensure_ascii=False, indent=2)

    print(f"{folder} done")