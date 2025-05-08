# Image path for the images
import pandas as pd
image_path = r"C:\DeepLearning\finalProject\flickr8k\Images"

# Read the CSV file containing captions
data = pd.read_csv(r"C:\DeepLearning\finalProject\flickr8k\captions.txt")

# Create a dictionary mapping image names to captions
def load_captions(data):
    mapping = {}
    for _, row in data.iterrows():
        image_id, caption = row['image'], row['caption']
        image_id = image_id.split('#')[0]  # Strip any extra info from image ID
        if image_id not in mapping:
            mapping[image_id] = []
        mapping[image_id].append(caption)
    return mapping