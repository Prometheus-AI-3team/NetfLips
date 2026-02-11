import pickle
import sys
import os
import shutil

def modify_bbox_pickle(pkl_path, modifications):
    """
    Modifies a bbox pickle file by setting specific frame ranges to None.
    
    Args:
        pkl_path (str): Path to the .bbox.pkl file.
        modifications (list of tuple): List of (start_frame, end_frame) tuples. 
                                       Frames in the range [start_frame, end_frame) will be set to None.
    """
    if not os.path.exists(pkl_path):
        print(f"Error: File not found at {pkl_path}")
        return

    # 1. Load the pickle file
    print(f"Loading {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        bbox_data = pickle.load(f)
    
    total_frames = len(bbox_data)
    print(f"Total frames: {total_frames}")

    # 2. Apply modifications
    modified_count = 0
    for start, end in modifications:
        # Clamp indices to valid range
        start = max(0, start)
        end = min(total_frames, end)
        
        if start >= end:
            print(f"Warning: Invalid range ({start}, {end}). Skipping.")
            continue
            
        print(f"Setting frames {start} to {end-1} -> None")
        for i in range(start, end):
            if bbox_data[i] is not None:
                bbox_data[i] = None
                modified_count += 1
    
    print(f"Total frames modified: {modified_count}")

    # 3. Create a backup
    backup_path = pkl_path + ".bak"
    shutil.copy2(pkl_path, backup_path)
    print(f"Backup created at {backup_path}")

    # 4. Save the modified data
    with open(pkl_path, 'wb') as f:
        pickle.dump(bbox_data, f)
    
    print(f"Successfully saved modified pickle to {pkl_path}")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Change these values to match your needs
    
    # Path to your .bbox.pkl file
    target_pickle_file = "/home/2022113135/gyucheol/NetfLips/data/final_bbox/hulk_h264_part2.bbox.pkl" 
    
    # List of ranges to set to None. Format: (start_index, end_index)
    # The end_index is exclusive (Python slice style).
    # Example: Set frames 100 to 109 to None -> (100, 110)
    ranges_to_set_none = [
        # (start_frame, end_frame),
        (41, 84),
        (161, 224)
    ]
    # ---------------------

    if target_pickle_file == "path/to/your/video.bbox.pkl":
        print("Please edit the script to specify the 'target_pickle_file' and 'ranges_to_set_none' first.")
    else:
        modify_bbox_pickle(target_pickle_file, ranges_to_set_none)
