import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from pathlib import Path
from PIL import Image

def get_dataloader(kwargs):
    dataset = Images(kwargs["input_dir"], duplicates=kwargs["duplicates"])
    dataloader = DataLoader(dataset, batch_size=kwargs["batch_size"])
    return dataloader

class Images(Dataset):
    def __init__(self, root_dir, duplicates):
        self.root_path = Path(root_dir)
        self.image_list = list(self.root_path.glob("*.png"))
        self.duplicates = duplicates # Number of times to duplicate the image in the dataset to produce multiple HR images

    def __len__(self):
        return self.duplicates*len(self.image_list)

    def generate_mask(self):
        mask = torch.zeros([1, 1024, 1024])
        mask[:,256:769, 256:769] = 1
        return mask


    def __getitem__(self, idx):
        img_path = self.image_list[idx//self.duplicates]
        image = torchvision.transforms.ToTensor()(Image.open(img_path))
        mask = self.generate_mask()
        image = image*(1-mask) + mask

        if(self.duplicates == 1):
            return image, mask, img_path.stem
        else:
            return image, mask, img_path.stem+f"_{(idx % self.duplicates)+1}"