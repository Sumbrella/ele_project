import os

import matplotlib.pyplot as plt

from common.unit import SingleFile

if __name__ == '__main__':
    father_dir = os.path.dirname(__file__)

    before_data_dir = os.path.join(father_dir, 'data/origin/before')
    after_data_dir = os.path.join(father_dir, 'data/origin/after')
    figures_dir = os.path.join(os.path.join(father_dir, 'data'), 'figures')
    if not os.path.exists(figures_dir):
        os.mkdir(figures_dir)

    for filename in os.listdir(before_data_dir):
        datafile = SingleFile(filepath=os.path.join(before_data_dir, filename))
        data_reader = datafile.get_reader(batch_size=10)

        single_file_path = os.path.join(figures_dir, f'{datafile.filename}')

        if not os.path.exists(single_file_path):
            os.mkdir(single_file_path)

        for point_id, points in enumerate(data_reader()):
            for point in points:
                plt.figure(figsize=(6, 8))
                point.plot(show=False)
                plt.savefig(
                    os.path.join(single_file_path, f'point_{point_id}.jpg')
                )
