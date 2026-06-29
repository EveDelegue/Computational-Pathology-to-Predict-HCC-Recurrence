import pandas as pd
import numpy as np


def main():
    legal_columns_to_display = [
        "Patient",
        "Nbre de lames",
        "Taille (cm)",
        "Valeur exacte AFP pré-opératoire",
        "Nombre de nodules",
        "Expansif multinodulaire",
        "Récidive avant 2 ans",
    ]

    dfs = pd.read_excel("data/tabs/table_prognosis.xlsx", sheet_name=None)

    df_pb = dfs[list(dfs.keys())[0]][legal_columns_to_display]
    df_hm = dfs[list(dfs.keys())[1]][legal_columns_to_display]
    df_bj = dfs[list(dfs.keys())[2]][legal_columns_to_display]

    temp = pd.concat([df_pb, df_hm], ignore_index=True)
    df_clin = pd.concat([temp, df_bj], ignore_index=True)

    df_clin[
        [
            "log1p_taille",
            "log1p_AFP",
        ]
    ] = df_clin[
        [
            "Taille (cm)",
            "Valeur exacte AFP pré-opératoire",
        ]
    ].apply(np.log1p)

    df_clin.rename(columns={"Récidive avant 2 ans": "Récidive Globale"}, inplace=True)
    df_clin.rename(columns={"Patient": "patient"}, inplace=True)

    df_tumor = pd.read_csv("data/tabs/final_tumor_features.csv")
    df_nuclear = pd.read_csv("data/tabs/final_nuclear_features.csv")
    df_inflams = pd.read_csv("data/tabs/final_inflams_features.csv")

    temp1 = pd.merge(df_clin, df_tumor, on="patient", how="inner")
    temp2 = pd.merge(df_nuclear, df_inflams, on="patient", how="inner")
    data = pd.merge(temp1, temp2, on="patient", how="inner")
    data.head()

    data.to_excel("data/tabs/input_dataframe_prognosis.xlsx")

if __name__ == "__main__":
    main()
