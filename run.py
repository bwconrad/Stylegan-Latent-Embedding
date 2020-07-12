import torch
from torch.nn import DataParallel
from pathlib import Path
from PIL import Image
import torchvision
from math import log10, ceil
import argparse

from src.PULSE import PULSE
from src.dataset import get_dataloader

parser = argparse.ArgumentParser(description='PULSE')

#I/O arguments
parser.add_argument('-input_dir', type=str, default='data/input_aligned', help='input data directory')
parser.add_argument('-output_dir', type=str, default='data/output', help='output data directory')
parser.add_argument('-cache_dir', type=str, default='cache', help='cache directory for model weights')
parser.add_argument('-duplicates', type=int, default=1, help='How many HR images to produce for every image in the input directory')
parser.add_argument('-batch_size', type=int, default=1, help='Batch size to use during optimization')

#PULSE arguments
parser.add_argument('-seed', type=int, help='manual seed to use')
parser.add_argument('-loss_str', type=str, default="100*L2+0.05*GEOCROSS", help='Loss function to use')
parser.add_argument('-eps', type=float, default=2e-3, help='Target for downscaling loss (L2)')
parser.add_argument('-noise_type', type=str, default='trainable', help='zero, fixed, or trainable')
parser.add_argument('-num_trainable_noise_layers', type=int, default=5, help='Number of noise layers to optimize')
parser.add_argument('-tile_latent', action='store_true', help='Whether to forcibly tile the same latent 18 times')
parser.add_argument('-bad_noise_layers', type=str, default="17", help='List of noise layers to zero out to improve image quality')
parser.add_argument('-opt_name', type=str, default='adam', help='Optimizer to use in projected gradient descent')
parser.add_argument('-learning_rate', type=float, default=0.4, help='Learning rate to use during optimization')
parser.add_argument('-steps', type=int, default=100, help='Number of optimization steps')
parser.add_argument('-lr_schedule', type=str, default='linear1cycledrop', help='fixed, linear1cycledrop, linear1cycle')
parser.add_argument('-save_intermediate', action='store_true', help='Whether to store and save intermediate HR and LR images during optimization')

kwargs = vars(parser.parse_args())

out_path = Path(kwargs["output_dir"])
out_path.mkdir(parents=True, exist_ok=True)

dataloader = get_dataloader(kwargs)

model = DataParallel(PULSE(cache_dir=kwargs["cache_dir"]))

toPIL = torchvision.transforms.ToPILImage()

for ref_im, ref_mask, ref_im_name in dataloader:
    if(kwargs["save_intermediate"]):
        padding = ceil(log10(100))
        for i in range(kwargs["batch_size"]):
            int_path_HR = Path(out_path / ref_im_name[i] / "HR")
            int_path_LR = Path(out_path / ref_im_name[i] / "LR")
            int_path_HR.mkdir(parents=True, exist_ok=True)
            int_path_LR.mkdir(parents=True, exist_ok=True)
        for j,(HR,LR) in enumerate(model(ref_im, ref_mask, **kwargs)):
            for i in range(kwargs["batch_size"]):
                toPIL(HR[i].cpu().detach().clamp(0, 1)).save(
                    int_path_HR / f"{ref_im_name[i]}_{j:0{padding}}.png")
                toPIL(LR[i].cpu().detach().clamp(0, 1)).save(
                    int_path_LR / f"{ref_im_name[i]}_{j:0{padding}}.png")
    else:
        for j,(output, masked_output) in enumerate(model(ref_im, ref_mask, **kwargs)):
            for i in range(kwargs["batch_size"]):
                # Save output
                toPIL(output[i].cpu().detach().clamp(0, 1)).save(
                    out_path / f"{ref_im_name[i]}.png")

                # Save masked output
                toPIL(masked_output[i].cpu().detach().clamp(0, 1)).save(
                    out_path / f"{ref_im_name[i]}_masked.png")
                
                merged_output = ref_im
                merged_output[i, :,256:769, 256:769] = output[i,:,256:769, 256:769]
                toPIL(merged_output[i].cpu().detach().clamp(0, 1)).save(
                    out_path / f"{ref_im_name[i]}_merged.png")
