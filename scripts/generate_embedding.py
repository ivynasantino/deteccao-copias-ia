from PIL import Image
from numpy import asarray, expand_dims
from matplotlib import pyplot
import pandas as pd
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
import os
from keras_vggface.utils import preprocess_input, decode_predictions
import time
import argparse

# extract a single face from a given photograph
def extract_face(filename, required_size=(224, 224)):
    pixels = pyplot.imread(filename)

    detector = MTCNN()
    results = detector.detect_faces(pixels)
    if results:
        # extract the bounding box from the first face
        x1, y1, width, height = results[0]["box"]
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
        return face_array
    else:
        return []


def generate_embeddings(filenames):
    faces = []
    for f in filenames:
        face = extract_face(f)
        if len(face) != 0:
            faces.append(face)

    # convert into an array of samples
    samples = asarray(faces, "float32")
    # prepare the face for the model, e.g. center pixels

    samples = preprocess_input(samples, version=2)
    # create a vggface model
    model = VGGFace(
        model="resnet50", include_top=False, input_shape=(224, 224, 3), pooling="avg"
    )
    # perform prediction
    yhat = model.predict(samples)

    return yhat


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aplicando técnicas de reconhecimento facial."
    )

    parser.add_argument(
        "-d", "--caminho_bd", required=True, help="Caminho do banco de imagens."
    )

    return parser.parse_args()


def save_names_db(filenames, path_names):
    dc_names = {"name": filenames}
    df = pd.DataFrame(dc_names)
    df.to_csv(path_names, index=False)


def access_db_extract(path_db, random=False, n_imgs=10):
    db_imgs = []
    if random:
        for i in range(n_imgs):
            db_imgs.append(
                random.choice(
                    [
                        (path_db + x)
                        for x in os.listdir(path_db)
                        if os.path.isfile(os.path.join(path_db, x))
                    ]
                )
            )
    else:
        db_imgs = [(path_db + f) for f in os.listdir(path_db)]

    return db_imgs


def save_embeddings_db(predicts):
    with open("../data/embeddings.csv", "w") as f:
        for row in predicts:
            for value in row:
                f.write(f"{value},")
            f.write("\n")

    f.close()


def run_generate_embeddings():
    args = parse_args()
    path_db = args.caminho_bd
    db_imgs = access_db_extract(path_db)

    start = time.time()
    predicts = generate_embeddings(db_imgs)
    end = time.time()
    total_time = end - start
    print("Gerar embeddings")
    print(f"Tempo total: {total_time} segundos")

    save_names_db(db_imgs, "../data/names_images.csv")
    save_embeddings_db(predicts)


if __name__ == "__main__":
    run_generate_embeddings()
