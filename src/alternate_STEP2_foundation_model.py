import torch
import os
import argparse
import yaml
from gigapath.pipeline_eve import run_inference_with_tile_encoder,load_tile_slide_encoder,run_inference_with_slide_encoder
import pandas as pd
import shutil



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide_name", type=str)
    parser.add_argument("--model", type=str,choices=["gigapath",'titan'])
    parser.add_argument("--xlsx", type=str)
    args = parser.parse_args()
    return args


def main():
    # parse arguments
    args = parse_arguments()
    slide_name = args.slide_name.split("/")[-1]
    model_name = args.model
    save_path="data/tmp"

    # read config file
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        features_checkpoints = config["paths"]["pth_to_features"]
        xlsx_path = args.xlsx
    # read the list of slides to process
    df0 = pd.read_excel(xlsx_path, sheet_name="PB").dropna(subset=["Patient"])
    df1 = pd.read_excel(xlsx_path, sheet_name="HMN").dropna(subset=["Patient"])
    df2 = pd.read_excel(xlsx_path, sheet_name="BJN").dropna(subset=["Patient"])
    list_patients0 = df0["Patient"].tolist()
    list_patients1 = df1["Patient"].tolist()
    list_patients2 = df2["Patient"].tolist()
    # check which hospital the slide belongs to
    if int(slide_name[:-1]) in list_patients0:
        hospital = "PB"
    elif int(slide_name[:-1]) in list_patients1:
        hospital = "HM"
    elif int(slide_name[:-1]) in list_patients2:
        hospital = "BJ"
    else:
        hospital = "Unknown"
    
    # only process if not already done
    # only process if in the list of slides to process
    if not os.path.exists(
        f"{features_checkpoints}/{slide_name}_features.pt"
    ) and hospital != "Unknown":  
    
        # set path
        patches_dir = config["paths"]["pth_to_patches"]
        # preprocess patches (not needed here because the foundation model includes normalization)
        """if hospital == "BJ":
            reference_pb = plt.imread("notebooks/HES__5.jpeg")
            mean_src, std_src = color_trans_cuda_fit(torch.tensor(reference_pb).permute(2,0,1).unsqueeze(0))
            new_slide_name = slide_name+"_norm"
            print("patches are from Beaujon ==> color transfer is needed")
            for patch_path in tqdm.tqdm(os.listdir(f"{patches_dir}/{slide_name}")):
                x = plt.imread(f"{patches_dir}/{slide_name}/{patch_path}")
                x_norm = color_trans_cuda_apply(torch.tensor(x).permute(2,0,1).unsqueeze(0), mean_src, std_src).squeeze(0).permute(1,2,0).numpy()
                os.makedirs(f"{patches_dir}/{new_slide_name}", exist_ok=True)
                plt.imsave(f"{patches_dir}/{new_slide_name}/{patch_path}", x_norm.astype(np.uint8))
            slide_name = new_slide_name
        """
        # load model
        tile_encoder,slide_encoder_model = load_tile_slide_encoder()
        if model_name == 'gigapath':
            with torch.no_grad():
                # run inference with the tile encoder
                list_patches = [os.path.join(f"{patches_dir}/{slide_name}", i) for i in os.listdir(f"{patches_dir}/{slide_name}") if i.endswith('.png') or i.endswith('.jpg')]
                tile_encoder_outputs = run_inference_with_tile_encoder(list_patches, tile_encoder)
                # run inference with the slide encoder
                slide_embeds = run_inference_with_slide_encoder(slide_encoder_model=slide_encoder_model, **tile_encoder_outputs)
                torch.save(slide_embeds,f"{features_checkpoints}/{slide_name.split('_')[0]}_features.pt")

        # if we have a features path, we can delete the patches to save space
        if os.path.exists(f"{features_checkpoints}/{slide_name.split('_')[0]}_features.pt"):
            shutil.rmtree(f"{patches_dir}/{slide_name}")

    else:
        print(slide_name, "already processed or not in the list to process")


if __name__ == "__main__":
    main()
