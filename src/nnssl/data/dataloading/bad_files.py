import os
import torch
import pickle
import numpy as np
from tqdm import tqdm

DATASET_DIR = "/path/to/dataset"
image_identifiers = [...]  # List of your dataset identifiers (e.g., from self.image_identifiers)

bad_files = []

def is_nan_or_inf(arr):
    """Check for NaN or Inf in numpy array."""
    return np.isnan(arr).any() or np.isinf(arr).any()

for img_id in tqdm(image_identifiers, desc="Checking files"):
    try:
        # Load case
        data, anon, anat, properties = your_dataset.load_case(DATASET_DIR, your_dataset.image_dataset, img_id)
        
        # Convert to torch to check NaN/Inf
        arr = np.asarray(data)
        if is_nan_or_inf(arr):
            print(f"[BAD DATA] NaN/Inf detected in: {img_id}")
            bad_files.append(img_id)

    except (RuntimeError, FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
        print(f"[CORRUPT] {img_id}: {e}")
        bad_files.append(img_id)

print("\nTotal bad files:", len(bad_files))
print("Bad file list:")
for f in bad_files:
    print(f)
