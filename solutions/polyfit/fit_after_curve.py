import os

import pandas as pd

from common.unit import SingleFile
from common.functions import fit_point

train_dir = "../../data/train"
test_dir = "../../data/test"


def fit_file(file_path, save_path, show=False):
    print('[INFO] FITTING FILE:{}'.format(file_path))
    file = SingleFile(file_path)
    path, file_name = os.path.split(file_path)
    file_name, ext = os.path.splitext(file_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_file = os.path.join(save_path, file_name + '.csv')

    all_params = []
    reader = file.get_reader(batch_size=1)

    for point_id, points in enumerate(reader()):
        for point in points:
            print('\t[INFO] FITTING POINT {}'.format(point_id))
            params = fit_point(point, show=show)
            print('\t[INFO] FITTED POINT {} RESULT:\n\t{}'.format(point_id, params))
            all_params.append(params)

    df = pd.DataFrame(all_params, columns=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'])
    df.to_csv(save_file)


def fit_dir(dir_path, save_path, show=False):
    files = os.listdir(dir_path)
    for file in files:
        try:
            file_path = os.path.join(dir_path, file)
            fit_file(file_path, save_path=save_path, show=show)
        except Exception as e:
            print(e)
            continue



if __name__ == '__main__':
    fit_dir(os.path.join(train_dir, 'after'), save_path=os.path.join(train_dir, 'teacher'), show=False)
    fit_dir(os.path.join(test_dir, 'after'), save_path=os.path.join(test_dir, 'teacher'), show=False)
    # fit_file("../../data/train/after/new_LINE_100_dbdt.dat", "../../data/train/after/teacher")
