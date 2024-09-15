import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from src.utils import download_images

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')

os.makedirs(IMAGE_DIR, exist_ok=True)

# Load the data
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

# Download images
print("Downloading training images...")
#download_images(train_df, IMAGE_DIR)
print("Downloading test images...")
#download_images(test_df, IMAGE_DIR)

# Define a custom dataset
class ProductImageDataset(Dataset):
    def _init_(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def _len_(self):
        return len(self.dataframe)

    def _getitem_(self, idx):
        img_name = os.path.join(self.image_dir, f"{self.dataframe.iloc[idx]['index']}.jpg")
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.dataframe.iloc[idx]['entity_value'] if 'entity_value' in self.dataframe.columns else ""
        
        return image, label

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = ProductImageDataset(train_df, IMAGE_DIR, transform=transform)
test_dataset = ProductImageDataset(test_df, IMAGE_DIR, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Print some information
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")

# Example of iterating through the data
for images, labels in train_loader:
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    break

print("Setup complete!")