''' Embed image into StyleGANs latent space '''

import torch
from torch.nn import DataParallel
from pathlib import Path
from torchvision.utils import save_image
import argparse
import pickle

from src.model import EmbeddingModel
from src.dataset import get_dataloader

parser = argparse.ArgumentParser()

# Paths arguments
parser.add_argument('-input_dir', type=str, default='data/input', help='Input data directory')
parser.add_argument('-output_dir', type=str, default='data/output', help='Output data directory')

# Model arguments
parser.add_argument('-batch_size', type=int, default=1, help='Batch size')
parser.add_argument('-seed', type=int, help='manual seed to use')
parser.add_argument('-loss_l2', type=float, default=1, help='L2 loss weight')
parser.add_argument('-loss_l1', type=float, default=0, help='L1 loss weight')
parser.add_argument('-loss_geocross', type=float, default=0, help='Geocross loss weight')
parser.add_argument('-noise_type', type=str, default='trainable', help='zero, fixed, or trainable')
parser.add_argument('-num_trainable_noise_layers', type=int, default=18, help='Number of noise layers to optimize')
parser.add_argument('-tile_latent', action='store_true', help='Whether to forcibly tile the same latent 18 times')
parser.add_argument('-bad_noise_layers', type=str, default="17", help='List of noise layers to zero out')
parser.add_argument('-optim', type=str, default='adam', help='Optimizer to use during optimization')
parser.add_argument('-learning_rate', type=float, default=0.4, help='Learning rate to use during optimization')
parser.add_argument('-steps', type=int, default=100, help='Number of optimization steps')
parser.add_argument('-lr_schedule', type=str, default='linear1cycledrop', help='fixed, linear1cycledrop, linear1cycle')
parser.add_argument('-save_intermediate', action='store_true', help='Whether to save intermediate images during optimization')
parser.add_argument('-save_latents', action='store_true', help='Whether to save the final latent and noise vectors to file')

kwargs = vars(parser.parse_args())

# Setup output paths
out_path = Path(kwargs["output_dir"])
out_path.mkdir(parents=True, exist_ok=True)

# Load data
dataloader = get_dataloader(kwargs['input_dir'], kwargs['batch_size'])

# Load model
model = DataParallel(EmbeddingModel(cache_dir='cache'))


# Run model on each image
for ref_im, ref_im_name in dataloader:
    if(kwargs["save_intermediate"]):
        # Create output directories for each image
        for i in range(kwargs["batch_size"]):
            int_path = Path(out_path / ref_im_name[i])
            int_path.mkdir(parents=True, exist_ok=True)
        
        # Save current image after each step 
        for j, (output, latent, noise) in enumerate(model(ref_im, **kwargs)):
            for i in range(kwargs["batch_size"]):
                save_image(output[i], int_path / f"{ref_im_name[i]}_{j:0{4}}.png")

    else:
        for j, (output, latent, noise) in enumerate(model(ref_im, **kwargs)):
            for i in range(kwargs["batch_size"]):
                # Save output
                save_image(output[i], f'{out_path}/{ref_im_name[i]}.png')

                # Save ground truth
                save_image(ref_im[i], f'{out_path}/{ref_im_name[i]}_gt.png')

                # Save final latent and noise vectors
                if kwargs['save_latents']:
                    noise = [n.detach().cpu() for n in noise]
                    with open(f'{out_path}/{ref_im_name[i]}_latents.pkl', 'wb') as f: 
                        pickle.dump([latent.detach().cpu(), noise], f)