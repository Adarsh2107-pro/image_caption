import os
from typing import Dict, List

import hydra
import pandas as pd


class DatasetLoader:
    def __init__(self, config: Dict):
        # Hydra changes the working directory -- change it back
        os.chdir(hydra.utils.get_original_cwd())

        # Automatically set when an object is created
        self.image_path: str = config['img_path'] # Set to relative path
        self.data: pd.DataFrame = pd.read_csv(config['captions_path']) # Set to relative path

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
