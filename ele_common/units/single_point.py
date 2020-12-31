import os
from typing import TextIO
from loguru import logger
import pandas as pd


class SinglePoint:

    def __init__(self, size, data):
        """
        :param size: the number of input
        :param data: [[x1, y1], [x2, y2], ...]
        """
        self._size = size
        self._data = data
        self._x = [data[i][0] for i in range(size)]
        self._y = [data[i][1] for i in range(size)]

    @classmethod
    def from_file(cls, fp: TextIO, skip_line=2):
        id = fp.readline()
        # print(id)
        if id is None:
            raise RuntimeError('No more line exist in {}'.format(fp))
        for i in range(skip_line):
            fp.readline()

        size = eval(fp.readline())
        data = []

        for i in range(size):
            now_position = fp.tell()
            line = fp.readline()
            try:
                x, y = map(float, line.split())
                data.append([x, y])
            except Exception as e:
                logger.warning(f"Except point number in {fp} is {size} but only get {i} points")
                # 拟合数据中的数据长度不等于self,_size
                fp.seek(now_position)  # 回到上一行
                size = i   # 更新点的数量
                break

        return SinglePoint(size, data)

    @property
    def size(self):
        return self._size

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def data(self):
        return self._data

    def plot(self, show=True, label='origin', alpha=1):
        import matplotlib.pyplot as plt
        df = pd.DataFrame(self._data)
        x = df.iloc[:, 0]
        y = df.iloc[:, 1]
        plt.plot(x, y, label=label, alpha=alpha)
        if show:
            plt.show()

    def get_data(self):
        return [self.x, self.y]

    def get_narray(self, dtype):
        import numpy as np
        return np.array([self.x, self.y], dtype=dtype)

    def add_to_file(self, full_path):
        if not os.path.exists(full_path):
            import time
            # 如果不存在文件，则创建文件并写入 0 和 日期
            fp = open(os.path.join(full_path), 'w')
            fp.write(' 0\n')
            fp.write(" ")
            fp.write(time.asctime())
            fp.write("\n")
        with open(os.path.join(full_path), 'r+') as fp:
            # 读取第一行的point数目，并令其+1
            point_number = int(fp.readline())
            fp.seek(0)
            fp.writelines(' {}\n'.format(point_number + 1))
        with open(os.path.join(full_path), 'a+') as fp:
            # 写入一个point
            fp.write(' {}\n'.format(point_number + 1))
            fp.write(' 1   2   3   4\n')
            fp.write(' xy\n')
            fp.write(' {}\n'.format(len(self.x)))
            for x, y in zip(self.x, self.y):
                fp.write(' {:e}\t\t{:e}\n'.format(x, y))


if __name__ == '__main__':
    # TODO: change the path
    fp = open('../../data/origin/before/LINE_100_dbdt.dat', 'r')
    fp.readline()
    fp.readline()
    sp = SinglePoint.from_file(fp)
    sp.plot()
