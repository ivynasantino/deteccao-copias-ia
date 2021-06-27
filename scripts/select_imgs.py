import random, os
import glob
import shutil
import argparse

def select_images(dst_path, num_imgs):
    random_filename = []
    for i in range(num_imgs):
        random_filename.append(
            random.choice(
                [
                    path + "/" + x
                    for x in os.listdir(path)
                    if os.path.isfile(os.path.join(path, x))
                ]
            )
        )
    print(random_filename)

    for p in random_filename:
        shutil.copy(p, dst_path)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Seleciona imagens de um dataset."
    )

    parser.add_argument(
        "-p", "--caminho_destino", required=True, help="Caminho de destino das imagens selecionadas."
    )
    parser.add_argument(
        "-n", "--numero_imagens", required=True, help="Número de imagens selecionadas."
    )

    return parser.parse_args()

def run_select():
    args = parse_args()
    dst_path = args.caminho_destino
    num_imgs = args.numero_imagens

    select_images(dst_path, num_imgs)

if __name__ == "__main__":
    run_select()
