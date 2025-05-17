from typing import Dict, List
import pandas as pd

class DatasetLoader:
    def __init__(self):
        # Automatically set when an object is created
        self.image_path: str = r"C:\GITproject\image_caption\flickr8k\Images"
        self.data: pd.DataFrame = pd.read_csv(r"C:\GITproject\image_caption\flickr8k\captions.txt")

    def load_captions(self) -> Dict[str, List[str]]:
        """
        Convert dataframe of captions to a dictionary mapping image_id -> list of captions
        """
        mapping: Dict[str, List[str]] = {}
        for _, row in self.data.iterrows():
            image_id, caption = row['image'], row['caption']
            image_id = image_id.split('#')[0]  # Remove extra info from image ID
            if image_id not in mapping:
                mapping[image_id] = []
            mapping[image_id].append(caption)
        return mapping
