import os
import pandas as pd
import yaml

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # load config
    nuc_features = config["nuc_features"]
    nuc_checkpoint = config["paths"]["pth_to_nuc_ckpts"]
    
    # create dataframe with each line is a slide
    df = pd.DataFrame(columns=["lame"] + nuc_features)
    df["lame"] = [e.split("_")[0] for e in os.listdir(nuc_checkpoint)]

    # add the mean of the parameters
    for data_name in os.listdir(nuc_checkpoint):
        data = pd.read_csv(os.path.join(nuc_checkpoint,data_name))
        slide_name = data_name.split("_")[0]
        df.loc[df["lame"] == slide_name] = [slide_name] + list(data[nuc_features].mean())

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
        df_nuclear.loc[df_nuclear["patient"] == patient] = [patient] + list(
            df.loc[df["patient"] == patient][nuc_features].mean()
        )

    # save in a tab
    df_nuclear.head()
    df_nuclear.to_csv(config["paths"]["pth_to_tab"]+'/final_nuclear_features.csv', index=False)

if __name__== "__main__":
    main()