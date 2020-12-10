import os

import pandas as pd
import numpy as np

from ele_common.units import SingleFile


def get_reader(data_dir, csv_dir, batch_size=10, debug=False, dtype='float32'):
    """

    :param data_dir: 数据文件夹
    :param csv_dir:  标签csv文件夹
    :param batch_size: 批大小
    :param debug:  模式
    :param dtype:  类型
    :returns: [N, C, W, H], [N, ]
    """

    def reader():
        points = []
        labels = []
        for file_name in os.listdir(data_dir):

            if debug:
                print('[INFO] getting teacher from {}'.format(file_name))

            file_path = os.path.join(data_dir, file_name)
            file_name, ext = os.path.splitext(file_name)
            if debug:
                print("[INFO] Reading file {}".format(file_name))
            try:
                file = SingleFile(file_path)
            except Exception as e:
                print(e)
                continue

            point_reader = file.get_reader(batch_size=1)
            label_df = pd.read_csv(os.path.join(csv_dir, 'new_' + file_name + '.csv'), index_col=0)
            # print(label_df)
            # print(label_df.iloc[0, :])
            for point_id, point in enumerate(point_reader()):
                point = point[0]
                label_series = label_df.iloc[point_id, :]
                points.append(point.get_data())
                labels.append(label_series.to_list())
                if len(points) is batch_size:
                    points = np.array(points, dtype).reshape(len(points), 2, 1, 100)
                    labels = np.array(labels, dtype).reshape(len(labels), -1)
                    yield points, labels
                    points = []
                    labels = []

        if len(labels):
            points = np.array(points, dtype).reshape(len(points), 2, 1, 100)
            labels = np.array(labels, dtype).reshape(len(labels), -1)
            yield points, labels


    return reader


if __name__ == '__main__':
    reader = get_reader("../../data/train/before", "../../data/train/teacher", batch_size=10, debug=True)
    for batch_id, data in enumerate(reader()):
        print(batch_id, data)