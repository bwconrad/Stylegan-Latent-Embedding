''' Reconstruct image from saved latent and noise vectors '''

import torch
from torch.nn import DataParallel
from torchvision.utils import save_image
import argparse
import pickle
from pathlib import Path
import os

from src.model import EmbeddingModel

parser = argparse.ArgumentParser()
parser.add_argument('-input', type=str, required=True, help='Input latent vector file')
parser.add_argument('-output_dir', type=str, default='data/output', help='Output data directory')
kwargs = vars(parser.parse_args())

# Setup output paths
out_path = Path(kwargs["output_dir"])
out_path.mkdir(parents=True, exist_ok=True)

# Load model
model = EmbeddingModel(cache_dir='cache')

# Load latent and noise vectors from file
with open(kwargs['input'], 'rb') as f:  
    latent, noise = pickle.load(f)

# Reconstruct into image
latent = latent.cuda()
noise = [n.cuda() for n  in noise]
output = (model.synthesis(latent, noise)+1)/2

# Save output
save_image(output, os.path.join(kwargs['output_dir'], 'output.png'))
