import os

import matplotlib.pyplot as plt

from common.unit import SingleFile

def generateAllPictures():
    father_dir = os.path.dirname(__file__)

    before_data_dir = os.path.join(father_dir, 'data/origin/before')
    after_data_dir = os.path.join(father_dir, 'data/origin/after')
    batch_size = 10

    figures_dir = os.path.join(os.path.join(father_dir, 'data'), 'figures')
    if not os.path.exists(figures_dir):
        os.mkdir(figures_dir)

    for filename in os.listdir(before_data_dir):
        datafile = SingleFile(filepath=os.path.join(before_data_dir, filename))
        data_reader = datafile.get_reader(batch_size=batch_size)

        single_file_path = os.path.join(figures_dir, f'{datafile.filename}')

        if not os.path.exists(single_file_path):
            os.mkdir(single_file_path)

        for batch_id, points in enumerate(data_reader()):
            for point_id, point in enumerate(points):
                point_id = batch_id * batch_size + point_id
                print(f'[INFO] drawing {filename}--point_{point_id}...')
                plt.figure()
                point.plot(show=False)
                plt.savefig(
                    os.path.join(single_file_path, f'point_{point_id}.jpg')
                )
                plt.close()

if __name__ == '__main__':
    generateAllPictures()