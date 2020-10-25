import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from pathlib import Path
from PIL import Image
import os

def get_dataloader(input_dir, batch_size):
    dataset = Images(input_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader

class Images(Dataset):
    def __init__(self, input_path):
        self.root_path = Path(input_path)
        if os.path.isdir(self.root_path):
            self.image_list = list(self.root_path.glob('*.png')) + list(self.root_path.glob('*.jpg'))
        else:
            self.image_list = [self.root_path]
        self.transforms = torchvision.transforms.Compose([
              torchvision.transforms.Resize(1024),
              torchvision.transforms.CenterCrop(1024),
              torchvision.transforms.ToTensor()  
            ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        image = self.transforms(Image.open(img_path).convert('RGB'))

        return image, img_path.stem
