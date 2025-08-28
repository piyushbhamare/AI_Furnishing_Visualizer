import os
import json
from collections import Counter
from tqdm import tqdm

def analyze_json_structures(json_folder):
    files = [f for f in os.listdir(json_folder) if f.endswith('.json')]
    total_files = len(files)

    structure_counter = Counter()
    error_files = []

    for json_file in tqdm(files, desc="Analyzing JSON files"):
        path = os.path.join(json_folder, json_file)
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            root_type = type(data).__name__

            if isinstance(data, dict):
                keys = tuple(sorted(data.keys()))
                structure_counter[f"dict_keys_{keys}"] += 1
            elif isinstance(data, list):
                structure_counter["list"] += 1
            else:
                structure_counter[f"other_{root_type}"] += 1
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
            error_files.append(json_file)

    print("\n--- JSON Structure Report ---")
    for struct, count in structure_counter.items():
        print(f"{struct}: {count} files")
    print(f"Files with errors: {len(error_files)}")
    if error_files:
        print("Error files:", error_files)

if __name__ == "__main__":
    annotation_folder = "/home/piyush/Desktop/AI Furnishing Visualizer/ADE20K_Dataset/training/ann"  # Adjust as needed
    analyze_json_structures(annotation_folder)
