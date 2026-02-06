import os
import pickle
import numpy as np
from tqdm import tqdm

TARGET_DIR = "/home/2022113135/gyucheol/NetfLips/data"

def convert_to_list(data):
    """
    Recursively convert numpy arrays to lists.
    """
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, list):
        return [convert_to_list(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(convert_to_list(item) for item in data)
    elif isinstance(data, dict):
        return {k: convert_to_list(v) for k, v in data.items()}
    else:
        return data

def main():
    if not os.path.exists(TARGET_DIR):
        print(f"Error: Directory {TARGET_DIR} does not exist.")
        return

    pkl_files = []
    for root, dirs, files in os.walk(TARGET_DIR):
        for file in files:
            if file.endswith(".pkl"):
                pkl_files.append(os.path.join(root, file))

    print(f"Found {len(pkl_files)} pickle files in {TARGET_DIR}")

    success_count = 0
    fail_count = 0

    for pkl_path in tqdm(pkl_files):
        try:
            # Load the data
            # Note: This requires the environment to have the SAME numpy version as the one that created it
            # if the file contains numpy arrays.
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)

            # Convert to list
            new_data = convert_to_list(data)

            # Save it back
            with open(pkl_path, 'wb') as f:
                pickle.dump(new_data, f)
            
            success_count += 1
            
        except Exception as e:
            print(f"Failed to process {pkl_path}: {e}")
            fail_count += 1

    print(f"\nProcessing complete.")
    print(f"Successfully converted: {success_count}")
    print(f"Failed: {fail_count}")

if __name__ == "__main__":
    main()
