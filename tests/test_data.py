import pytest
import yaml

from image_caption.data.exploring_dataset import DatasetLoader

# Define the dataset path
with open("config/conf/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Define the expected number of training and test samples
N_images = 8091

# The tutorial uses train and test datasets, but we can just check the single flickr8k dataset
@pytest.fixture
def loader():
    return DatasetLoader(cfg)

def test_dataset_length(loader):
    assert len(loader) == N_images

def test_captions_length(loader):
    assert len(loader.load_captions()) == N_images

### flickr8k dataset does not have a standard shape / size
# def test_data_shape(dataset):
#     for data, _ in dataset:
#         assert data.shape == (1, 28, 28) or data.shape == (784,)
