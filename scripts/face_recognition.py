from scipy.spatial.distance import cosine
import sys
import time
import numpy as np
from utils import get_embeddings, get_names
from operator import itemgetter

# determine if a candidate face is a match for a known face
def is_match_baseline(embeddings, name):
    start_all = time.time()
    copys = []
    for i in range(len(embeddings)):
        known_embedding = embeddings[i][0:-1]
        for j in range(len(embeddings)):
            if name[i] != name[j]:
                distance = cosine(known_embedding, embeddings[j][0:-1])
                if distance == 0.0:
                    copy = sorted((name[i][0], name[j][0]))
                    if copy not in copys:
                        copys.append(copy)
    end_all = time.time()
    total_time = end_all - start_all
    print(f"Tempo total para buscar cópias: {total_time} segundos")
    print(copys)


def is_match(embeddings, names, index):
    if index:
        print("Método não implementado.")
    else:
        is_match_baseline(embeddings, names)


def run_recognition():
    embeddings = get_embeddings()
    names = get_names()
    np.savetxt(sys.stdout, embeddings, delimiter=",", newline="\n")
    is_match(embeddings, names, index=False)


if __name__ == "__main__":
    run_recognition()
