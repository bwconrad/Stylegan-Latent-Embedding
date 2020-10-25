import torch
from torch.nn import DataParallel
from pathlib import Path
from PIL import Image
import torchvision
import argparse

from src.model import EmbeddingModel
from src.dataset import get_dataloader

parser = argparse.ArgumentParser()

# Paths arguments
parser.add_argument('-input_dir', type=str, default='data/input_aligned', help='input data directory')
parser.add_argument('-output_dir', type=str, default='data/output', help='output data directory')

# Model arguments
parser.add_argument('-batch_size', type=int, default=1, help='Batch size to use during optimization')
parser.add_argument('-seed', type=int, help='manual seed to use')
parser.add_argument('-loss_l2', type=float, default=1, help='L2 loss weight')
parser.add_argument('-loss_l1', type=float, default=0, help='L1 loss weight')
parser.add_argument('-loss_geocross', type=float, default=0, help='Geocross loss weight')
parser.add_argument('-noise_type', type=str, default='trainable', help='zero, fixed, or trainable')
parser.add_argument('-num_trainable_noise_layers', type=int, default=5, help='Number of noise layers to optimize')
parser.add_argument('-tile_latent', action='store_true', help='Whether to forcibly tile the same latent 18 times')
parser.add_argument('-bad_noise_layers', type=str, default="17", help='List of noise layers to zero out to improve image quality')
parser.add_argument('-optim', type=str, default='adam', help='Optimizer to use in projected gradient descent')
parser.add_argument('-learning_rate', type=float, default=0.4, help='Learning rate to use during optimization')
parser.add_argument('-steps', type=int, default=100, help='Number of optimization steps')
parser.add_argument('-lr_schedule', type=str, default='linear1cycledrop', help='fixed, linear1cycledrop, linear1cycle')
parser.add_argument('-save_intermediate', action='store_true', help='Whether to store and save intermediate images during optimization')

kwargs = vars(parser.parse_args())

# Setup output paths
out_path = Path(kwargs["output_dir"])
out_path.mkdir(parents=True, exist_ok=True)

# Load data
dataloader = get_dataloader(kwargs['input_dir'], kwargs['batch_size'])

# Load model
model = DataParallel(EmbeddingModel(cache_dir='cache'))

toPIL = torchvision.transforms.ToPILImage()

# Run model on each image
for ref_im, ref_im_name in dataloader:
    if(kwargs["save_intermediate"]):
        # Create output directories for each image
        for i in range(kwargs["batch_size"]):
            int_path = Path(out_path / ref_im_name[i])
            int_path.mkdir(parents=True, exist_ok=True)
        
        # Save current image after each step 
        for j, (output, _, _) in enumerate(model(ref_im, **kwargs)):
            for i in range(kwargs["batch_size"]):
                toPIL(output[i].cpu().detach().clamp(0, 1)).save(
                    int_path / f"{ref_im_name[i]}_{j:0{100}}.png")
            # Save ground truth
            toPIL(ref_im[i].cpu().detach().clamp(0, 1)).save(
                  '{}/{}_gt.png'.format(out_path, ref_im_name[i]))   
    else:
        for j, (output, _, _) in enumerate(model(ref_im, **kwargs)):
            for i in range(kwargs["batch_size"]):
                # Save output
                toPIL(output[i].cpu().detach().clamp(0, 1)).save(
                    '{}/{}.png'.format(out_path, ref_im_name[i]))

                # Save ground truth
                toPIL(ref_im[i].cpu().detach().clamp(0, 1)).save(
                    '{}/{}_gt.png'.format(out_path, ref_im_name[i]))
