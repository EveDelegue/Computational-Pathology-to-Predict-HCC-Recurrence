import os
import pandas as pd
import argparse
import yaml

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose",type=bool,default=True)
    parser.add_argument("--seuil_nb_lames",type=int,default=1000)
    args = parser.parse_args()
    return args

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # parse arguments
    args = parse_arguments()
    seuil_lames = args.seuil_nb_lames # TODO à faire
    # load config
    nuc_features = config["nuc_features"]
    nuc_checkpoint = config["paths"]["pth_to_nuc_ckpts"]
    
    # create dataframe where each line is a slide
    df = pd.DataFrame(columns=["lame"] + nuc_features + ['nb_patch'])
    df["lame"] = [e.split("_")[0] for e in os.listdir(nuc_checkpoint)]

    # add the sum of the parameters and the number of patchs
    for data_name in os.listdir(nuc_checkpoint):
        data = pd.read_csv(os.path.join(nuc_checkpoint,data_name))
        slide_name = data_name.split("_")[0]
        df.loc[df["lame"] == slide_name] = [slide_name] + list(data[nuc_features].sum()) + [len(data)]

    # add patient name
    df["patient"] = df["lame"].apply(lambda x: x[:-1]).astype(int)

    # create a dataframe where each line is a patient
    df_nuclear = pd.DataFrame(
        index=df["patient"].unique(),
        columns=["patient"] + nuc_features,
    )

    df_nuclear["patient"] = df["patient"].unique()

    # add the mean of the parameters
    for patient in df["patient"].unique():
        # sum the features for each slide
        sum_features = df.loc[df["patient"] == patient][nuc_features].sum()
        # sum the number of patchs for each slide
        sum_patchs = df.loc[df["patient"] == patient]['nb_patch'].sum()
        # the mean for the patient is the ratio between total sum of features and total number of patchs
        df_nuclear.loc[df_nuclear["patient"] == patient] = [patient] + list(
            sum_features/sum_patchs
        )

    # save in a tab
    df_nuclear.head()
    df_nuclear.to_csv(config["paths"]["pth_to_tab"]+'/final_nuclear_features.csv', index=False)

if __name__== "__main__":
    main()