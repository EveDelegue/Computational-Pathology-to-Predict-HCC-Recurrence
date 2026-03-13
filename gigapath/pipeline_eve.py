# --------------------------------------------------------
# Pipeline for running with GigaPath
# --------------------------------------------------------
import os
import timm
import torch
import shutil
import numpy as np
import pandas as pd
import gigapath.slide_encoder as slide_encoder

from tqdm import tqdm
from PIL import Image
from pathlib import Path
from torchvision import transforms
from typing import List, Tuple, Union
from torch.utils.data import Dataset, DataLoader
#from gigapath.preprocessing.data.create_tiles_dataset import process_slide


class TileEncodingDataset(Dataset):
    """
    Do encoding for tiles

    Arguments:
    ----------
    image_paths : List[str]
        List of image paths, each image is named with its coordinates
        Example: ['images/256x_256y.png', 'images/256x_512y.png']
    transform : torchvision.transforms.Compose
        Transform to apply to each image
    """
    def __init__(self, image_paths: List[str], transform=None):
        self.transform = transform
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = os.path.basename(img_path)
        # get x, y coordinates from the image name
        if ".jpg" in img_name:
            x, y = img_name.split('.jpg')[0].split('_')[2], img_name.split('.jpg')[0].split('_')[4]
        elif ".png" in img_name:
            x, y = img_name.split('.png')[0].split('_')[2], img_name.split('.png')[0].split('_')[4]
        x, y = int(x), int(y)
        # load the image
        with open(img_path, "rb") as f:
            img = Image.open(f).convert("RGB")
            if self.transform:
                img = self.transform(img)
        return {'img': torch.from_numpy(np.array(img)),
                'coords': torch.from_numpy(np.array([x, y])).float()}

'''
def tile_one_slide(slide_file:str='', save_dir:str='', level:int=0, tile_size:int=256):
    """
    This function is used to tile a single slide and save the tiles to a directory.
    -------------------------------------------------------------------------------
    Warnings: pixman 0.38 has a known bug, which produces partial broken images.
    Make sure to use a different version of pixman.
    -------------------------------------------------------------------------------

    Arguments:
    ----------
    slide_file : str
        The path to the slide file.
    save_dir : str
        The directory to save the tiles.
    level : int
        The magnification level to use for tiling. level=0 is the highest magnification level.
    tile_size : int
        The size of the tiles.
    """
    slide_id = os.path.basename(slide_file)
    # slide_sample = {"image": slide_file, "slide_id": slide_id, "metadata": {'TP53': 1, 'Diagnosis': 'Lung Cancer'}}
    slide_sample = {"image": slide_file, "slide_id": slide_id, "metadata": {}}

    save_dir = Path(save_dir)
    if save_dir.exists():
        print(f"Warning: Directory {save_dir} already exists. ")

    print(f"Processing slide {slide_file} at level {level} with tile size {tile_size}. Saving to {save_dir}.")
    # optional segmentation of the slide
    xml_segmentation_path = None
    #if slide_file.endswith('.svs'):
    #    xml_segmentation_path = slide_file.replace('.svs', '.xml')
    #    if not os.path.exists(xml_segmentation_path):
    #       print(f"Warning: XML segmentation file {xml_segmentation_path} does not exist. Skipping segmentation.")
    #        #xml_segmentation_path = None
    #else:
    #    print(f"Warning: Unsupported slide file format {slide_file}. Only .svs, are supported. Skipping segmentation.")
    # add xml segmentation path to slide sample if it exists    
    if xml_segmentation_path and os.path.exists(xml_segmentation_path):
        slide_sample["xml_segmentation"] = xml_segmentation_path
        print(f"Using XML segmentation file {xml_segmentation_path} for tiling.")
    else:
        print(f"No XML segmentation file found for {slide_file}. Tiling without segmentation.")
    slide_dir = process_slide(
        slide_sample,
        level=level,
        margin=0,
        tile_size=tile_size,
        foreground_threshold=None,
        occupancy_threshold=0.1,
        output_dir=save_dir / "output",
        thumbnail_dir=save_dir / "thumbnails",
        tile_progress=True,
    )

    dataset_csv_path = slide_dir / "dataset.csv"
    dataset_df = pd.read_csv(dataset_csv_path)
    assert len(dataset_df) > 0
    failed_csv_path = slide_dir / "failed_tiles.csv"
    failed_df = pd.read_csv(failed_csv_path)
    assert len(failed_df) == 0

    print(f"Slide {slide_file} has been tiled. {len(dataset_df)} tiles saved to {slide_dir}.")
'''

def load_tile_encoder_transforms() -> transforms.Compose:
    """Load the transforms for the tile encoder"""
    transform = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return transform


def load_tile_slide_encoder(local_tile_encoder_path: str='',
                            local_slide_encoder_path: str='',
                            global_pool=True) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """Load the GigaPath tile and slide encoder models.
    Note: Older versions of timm have compatibility issues.
    Please ensure that you use a newer version by running the following command: pip install timm>=1.0.3.
    """
    if local_tile_encoder_path:
        tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=False, checkpoint_path=local_tile_encoder_path)
    else:
        tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    print("Tile encoder param #", sum(p.numel() for p in tile_encoder.parameters()))

    if local_slide_encoder_path:
        slide_encoder_model = slide_encoder.create_model(local_slide_encoder_path, "gigapath_slide_enc12l768d", 1536, global_pool=global_pool)
    else:
        slide_encoder_model = slide_encoder.create_model("hf_hub:prov-gigapath/prov-gigapath", "gigapath_slide_enc12l768d", 1536, global_pool=global_pool)
    print("Slide encoder param #", sum(p.numel() for p in slide_encoder_model.parameters()))

    return tile_encoder, slide_encoder_model


@torch.no_grad()
def run_inference_with_tile_encoder(image_paths: List[str], tile_encoder: torch.nn.Module, batch_size: int=128,device= 'cuda', norm = lambda x: x, transform_daug=lambda x: x)   -> dict:
    """
    Run inference with the tile encoder

    Arguments:
    ----------
    image_paths : List[str]
        List of image paths, each image is named with its coordinates
    tile_encoder : torch.nn.Module
        Tile encoder model
    """
    if device == 'cuda':
        tile_encoder = tile_encoder.cuda()
    #tile_encoder = tile_encoder.cuda()
    # make the tile dataloader
    tile_dl = DataLoader(TileEncodingDataset(image_paths, transform=transforms.Compose([norm,load_tile_encoder_transforms(), transform_daug])), batch_size=batch_size, shuffle=False)
    # run inference
    tile_encoder.eval()
    collated_outputs = {'tile_embeds': [], 'coords': []}
    with torch.amp.autocast('cuda',dtype=torch.float16):
        for batch in tqdm(tile_dl):#tqdm(tile_dl, desc='Running inference with tile encoder'):
            if device == 'cuda':
                collated_outputs['tile_embeds'].append(tile_encoder(batch['img'].cuda()).detach().cpu())
            #collated_outputs['tile_embeds'].append(tile_encoder(batch['img'].cuda()).detach().cpu())
            else:
                collated_outputs['tile_embeds'].append(tile_encoder(batch['img']).detach().cpu())
            collated_outputs['coords'].append(batch['coords'])
    return {k: torch.cat(v) for k, v in collated_outputs.items()}


@torch.no_grad()
def run_inference_with_slide_encoder(tile_embeds: torch.Tensor, coords: torch.Tensor, slide_encoder_model: torch.nn.Module,device =   'cuda') -> torch.Tensor:
    """
    Run inference with the slide encoder

    Arguments:
    ----------
    tile_embeds : torch.Tensor
        Tile embeddings
    coords : torch.Tensor
        Coordinates of the tiles
    slide_encoder_model : torch.nn.Module
        Slide encoder model
    """
    if len(tile_embeds.shape) == 2:
        tile_embeds = tile_embeds.unsqueeze(0)
        coords = coords.unsqueeze(0)
    if device == 'cuda':
        #tile_embeds = tile_embeds.cuda()
        slide_encoder_model = slide_encoder_model.cuda()
    slide_encoder_model.eval()
    # run inference
    with torch.amp.autocast('cuda',dtype=torch.float16):
        if device == 'cuda':
            slide_embeds = slide_encoder_model(tile_embeds.cuda(), coords.cuda(), all_layer_embed=True)
        else:
           #slide_embeds = slide_encoder_model(tile_embeds.cuda(), coords.cuda(), all_layer_embed=True)
            slide_embeds = slide_encoder_model(tile_embeds, coords, all_layer_embed=True)
    outputs = {"layer_{}_embed".format(i): slide_embeds[i].cpu() for i in range(len(slide_embeds))}
    outputs["last_layer_embed"] = slide_embeds[-1].cpu()
    return outputs
