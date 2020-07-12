import glob
import dlib
from src.utils import open_url
from pathlib import Path
import argparse
from src.shape_predictor import align_face

parser = argparse.ArgumentParser(description='PULSE')

parser.add_argument('-input_dir', type=str, default='data/preprocessed', help='directory with unprocessed images')
parser.add_argument('-output_dir', type=str, default='data/input', help='output directory')
parser.add_argument('-output_size', type=int, default=32, help='size to downscale the input images to, must be power of 2')
parser.add_argument('-seed', type=int, help='manual seed to use')
parser.add_argument('-cache_dir', type=str, default='cache', help='cache directory for model weights')

args = parser.parse_args()

cache_dir = Path(args.cache_dir)
cache_dir.mkdir(parents=True, exist_ok=True)

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True,exist_ok=True)

print("Downloading Shape Predictor")
f=open_url("https://drive.google.com/uc?id=1huhv8PYpNNKbGCLOaYUjOgR1pY5pmbJx", cache_dir=cache_dir, return_path=True)
predictor = dlib.shape_predictor(f)

for im in Path(args.input_dir).glob("*.*"):
    faces = align_face(str(im),predictor)

    for i,face in enumerate(faces):
        face.save(Path(args.output_dir) / (im.stem+f"_{i}.png"))
