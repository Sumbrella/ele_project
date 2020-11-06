import os

import pandas as pd

from common.unit import SingleFile


def get_reader(data_dir, csv_dir, batch_size=10, debug=False):

    def reader():
        points = []
        labels = []
        for file_name in os.listdir(data_dir):
            if debug:
                print('[INFO] getting data from {}'.format(file_name))
            file_path = os.path.join(data_dir, file_name)
            file_name, ext = os.path.splitext(file_name)
            file = SingleFile(file_path)
            point_reader = file.get_reader(batch_size=1)
            label_df = pd.read_csv(os.path.join(csv_dir, 'new_' + file_name + '.csv'), index_col=0)
            # print(label_df)
            # print(label_df.iloc[0, :])
            for point_id, point in enumerate(point_reader()):
                point = point[0]
                label_series = label_df.iloc[point_id, :]
                points.append(point.get_narray(dtype='float32'))
                labels.append(label_series.to_numpy(dtype='float32'))
                if len(points) is batch_size:
                    yield points, labels
                    points = []
                    labels = []

        if len(labels):
            yield points, labels


    return reader


if __name__ == '__main__':
    reader = get_reader("../../data/train/before", "../../data/train/teacher", batch_size=10, debug=True)
    for batch_id, data in enumerate(reader()):
        print(batch_id, data)