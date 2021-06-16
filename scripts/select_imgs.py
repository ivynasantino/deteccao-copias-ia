import random, os
import glob
import shutil

path = "../../../../Downloads/img_align_celeba"
dst_path = "../db/baseline/"

random_filename = []
for i in range(10):
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
