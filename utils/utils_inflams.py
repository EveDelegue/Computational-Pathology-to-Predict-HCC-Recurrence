import joblib
import numpy as np
from scipy.spatial import distance
from tqdm.notebook import tqdm

# defining the coloring dictionary: a dictionary that specifies a color to each class {type_id : (type_name, colour)}
color_dict_pannuke = {
    0: ("background", (255, 255, 0)),  # YELLOW
    1: ("neoplastic epithelial", (255, 255, 0)),  # YELLOW
    2: ("Inflammatory", (0, 191, 255)),  # RED
    3: ("Connective", (255, 255, 0)),  # YELLOW
    4: ("Dead", (255, 255, 0)),  # YELLOW
    5: ("non-neoplastic epithelial", (255, 255, 0)),  # YELLOW
}

color_dict_monusac = {
    0: ("Epithelial", (255, 255, 0)),  # YELLOW
    1: ("Lymphocyte", (0, 191, 255)),  # green
    2: ("Macrophage", (0, 191, 255)),  # green
    3: ("Neutrophil", (255, 255, 0)),  # YELLOW
    4: ("Inflammatory", (0, 191, 255)),  # RED
}

color_dict_consep = {
    0: ("background", (255, 255, 0)),  # YELLOW
    1: ("Epithelial", (255, 255, 0)),  # YELLOW
    2: ("Inflammatory", (0, 191, 255)),  # RED
    3: ("Spindle-Shaped", (255, 255, 0)),  # YELLOW
    4: ("Miscellaneous", (255, 255, 0)),  # YELLOW
}

color_dict_AllInflams = {
    1: ("Lymphocyte", (0, 191, 255)),  # RED
    4: ("Inflammatory", (0, 191, 255)),
    2: ("Macrophage", (0, 191, 255)),
}


def get_Inflammatory(tile_preds, color_dict):
    """
    This function filters out nuclei that are classified as either 'Inflammatory' or 'Lymphocyte' based on the given
    tile predictions and color dictionary. It also counts the number of such nuclei.

    Parameters:
    tile_preds (dict): A dictionary containing nucleus predictions. Each key is a nucleus ID, and the corresponding value is
                        another dictionary containing the nucleus's properties, including its type.
    color_dict (dict): A dictionary mapping class labels to their corresponding descriptions and colors.

    Returns:
    tuple: A tuple containing two elements:
        - Inflammatory_nucleus (dict): A dictionary containing only the nuclei that are classified as 'Inflammatory' or
                                    'Lymphocyte'. Each key is a nucleus ID, and the corresponding value is the nucleus's
                                    properties.
        - number (int): The total number of 'Inflammatory' or 'Lymphocyte' nuclei found.
    """
    number = 0
    Inflammatory_nucleus = {}
    for nuc_id in tile_preds:
        class_label = tile_preds[nuc_id]["type"]
        classe = color_dict[class_label][0]
        if classe in ["Inflammatory", "Lymphocyte", "Macrophage"]:
            number += 1
            Inflammatory_nucleus[nuc_id] = tile_preds[nuc_id]
    return Inflammatory_nucleus, number


def filter_centroids(centroids, d_threshold=1.5):
    # Define distance threshold
    d_threshold = 1.5  # Adjust this value based on your data
    # List to store the filtered centroids
    filtered_centroids = []
    # Flag array to mark centroids that have already been clustered
    visited = np.zeros(len(centroids), dtype=bool)
    for i, centroid in enumerate(tqdm(centroids)):
        if visited[i]:
            continue
        # Group of centroids that are within d_threshold of the current centroid
        cluster = [centroid]
        # Mark the current centroid as visited
        visited[i] = True
        for j, other_centroid in enumerate(centroids[i + 1 :], start=i + 1):
            if (
                not visited[j]
                and distance.euclidean(centroid, other_centroid) <= d_threshold
            ):
                cluster.append(other_centroid)
                visited[j] = True
        # Choose the "big" centroid to keep; here we take the one with the largest sum of coordinates
        # You can adjust this to your criteria, like averaging the coordinates instead
        big_centroid = max(cluster, key=lambda x: sum(x))
        filtered_centroids.append(big_centroid)
    return filtered_centroids


def detect_Inflammatory_cells(img_path, models, colors):
    dict4 = {}
    for model, color in zip(models, colors):
        tile_output = model.predict([img_path], mode="tile")
        tile_preds = joblib.load(f"{tile_output[0][1]}.dat")
        Inflammatory_nucleus1, _ = get_Inflammatory(tile_preds, color)
        dict4.update(Inflammatory_nucleus1)
    nuc_centroids = {
        tuple(np.uint8(info["centroid"])): nuc_id for nuc_id, info in dict4.items()
    }

    filtered_centroids = np.array(
        filter_centroids(np.array(list(nuc_centroids.keys())), d_threshold=1.5)
    )
    filtred_dict = {}
    for centroid in filtered_centroids:
        nuc_id = nuc_centroids[tuple(centroid)]
        filtred_dict[nuc_id] = dict4[nuc_id]
    return filtred_dict
