# StyleGAN Latent Embedding Image Reconstruction
<p align="center">
  <img src="samples/tigger.gif", width="250" />
  <img src="samples/waterfall.gif", width="250" />
  <img src="samples/toast.gif", width="250" />
</p>

Embed any image into StyleGAN's latent space. Based off of [PULSE's](https://github.com/adamian98/pulse) optimization method and codebase.

## How To Run
### Requirements
Run ```pip install -r requirements.txt``` to install all required packages.
### Usage
Run ```python run.py -input_dir <PATH_TO_INPUT_DIR>``` to embed and save all images in the input
directory.
- ```-save_intermediate```: save the current image after each optimization step.
- ```-save_latent```: save the final latent and noise vector to file which can be used by
```decode.py```.

```embed.ipynb``` offers an interface to embed single images and visualize the reconstruction error.

Run ```python decode.py -input <LATENT_FILE>``` to reconstruct the image of a given saved latent and
noise vector from using the ```-save_latent``` flag with ```run.py```.
