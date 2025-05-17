from typing import Dict, List

import pandas as pd

# Josh comment: why do we need this line?
# image_path = r"C:\DeepLearning\finalProject\data\Images"

# Read the CSV file containing captions
# Josh comments:
    # we need to make this path relative
    # but also, why do we even need this lines?
# data = pd.read_csv(r"data/captions.txt")

# Create a dictionary mapping image names to captions
def load_captions(data: pd.DataFrame) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for _, row in data.iterrows():
        image_id, caption = row['image'], row['caption']
        image_id = image_id.split('#')[0]  # Strip any extra info from image ID
        if image_id not in mapping:
            mapping[image_id] = []
        mapping[image_id].append(caption)
    return mapping
