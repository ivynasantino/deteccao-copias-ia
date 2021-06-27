from scipy.spatial.distance import cosine
import sys
import time
import numpy as np
from utils import get_embeddings, get_names
from operator import itemgetter
import pandas as pd

# determine if a candidate face is a match for a known face
def is_match_baseline(embeddings, name):
    start_all = time.time()
    copys = []
    original = []
    copy = []
    for i in range(len(embeddings)):
        known_embedding = embeddings[i][0:-1]
        for j in range(len(embeddings)):
            if name[i] != name[j]:
                distance = cosine(known_embedding, embeddings[j][0:-1])
                if distance == 0.0:
                    cp = sorted((name[i][0], name[j][0]))
                    if cp not in copys:
                        copys.append(cp)
                        copy.append(cp[0])
                        original.append(cp[1])
    end_all = time.time()
    total_time = end_all - start_all
    print(f"Tempo total para buscar cópias: {total_time} segundos")
    dc_result = {"name_original": original, "name_copy": copy}
    df = pd.DataFrame(dc_result)
    df.to_csv("../data/results_10_index.csv", index=False)


# determine if a candidate face is a match for a known face
def is_match_index(embeddings, name):
    start_all = time.time()
    copys = []
    original = []
    copy = []
    for i in range(len(embeddings)):
        known_embedding = embeddings[i][0:-1]
        for j in range(len(embeddings)):
            if name[i][0] != name[j][0] and name[i][1] == name[j][1]:
                distance = cosine(known_embedding, embeddings[j][0:-1])
                if distance == 0.0:
                    cp = sorted((name[i][0], name[j][0]))
                    if cp not in copys:
                        copys.append(cp)
                        copy.append(cp[0])
                        original.append(cp[1])
    end_all = time.time()
    total_time = end_all - start_all
    print(f"Tempo total para buscar cópias: {total_time} segundos")
    dc_result = {"name_original": original, "name_copy": copy}
    df = pd.DataFrame(dc_result)
    df.to_csv("../data/results_copy_index_50.csv", index=False)


def is_match(embeddings, names, index):
    if index:
        is_match_index(embeddings, names)
    else:
        is_match_baseline(embeddings, names)


def run_recognition():
    embeddings = get_embeddings("../data/embeddings_copy_index_50.csv")
    names = get_names("../data/names_images_copy_index_50.csv")
    np.savetxt(sys.stdout, embeddings, delimiter=",", newline="\n")
    is_match(embeddings, names, index=True)


if __name__ == "__main__":
    run_recognition()
