import numpy as np
import pandas as pd


def save_predict_embeddings_db(dc_embeddings):
    df = pd.DataFrame(dc_embeddings)
    df.to_csv("../data/similar_images.csv", index=False)


def get_embeddings(path_db="../data/embeddings.csv"):
    embeddings = np.genfromtxt(path_db, delimiter=",")
    return embeddings


def get_names(path_db="../data/names_images.csv"):
    df = pd.read_csv(path_db)
    return df.values
