import os

import matplotlib.pyplot as plt

from common.unit import SingleFile


def generateAllPictures(data_dir):
    father_dir = os.path.dirname(data_dir)

    before_data_dir = os.path.join(data_dir, 'before')
    after_data_dir = os.path.join(data_dir, 'after')
    batch_size = 10

    figures_dir = os.path.join(father_dir, 'figures')

    if not os.path.exists(figures_dir):
        os.mkdir(figures_dir)

    for path_dir in [before_data_dir, after_data_dir]:
        for filename in os.listdir(path_dir):
            filepath = os.path.join(path_dir, filename)
            datafile = SingleFile(filepath=filepath)
            data_reader = datafile.get_reader(batch_size=batch_size)

            if path_dir is before_data_dir:
                single_file_path = os.path.join(figures_dir, 'before', f'{datafile.filename}')
            else:
                single_file_path = os.path.join(figures_dir, 'after', f'{datafile.filename}')

            if not os.path.exists(single_file_path):
                os.makedirs(single_file_path)

            for batch_id, points in enumerate(data_reader()):
                for point_id, point in enumerate(points):
                    point_id = batch_id * batch_size + point_id
                    print(f'[INFO] drawing {filename}--point_{point_id}...')
                    plt.figure()
                    # plt configs
                    plt.ylim(0, 30 * 1e-8)

                    point.plot(show=False)

                    plt.savefig(
                        os.path.join(single_file_path, f'point_{point_id}.jpg')
                    )
                    plt.close()


if __name__ == '__main__':
    generateAllPictures("data/origin")