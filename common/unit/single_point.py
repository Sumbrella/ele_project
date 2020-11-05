from typing import TextIO

import pandas as pd


class SinglePoint:

    def __init__(self, fp:TextIO, skip_line=2):
        id = fp.readline()
        if id is None:
            raise RuntimeError('No more line exist in {}'.format(fp))
        self._id = int(id)

        for i in range(skip_line):
            fp.readline()

        self._size = eval(fp.readline())
        self._data = []
        self._x = []
        self._y = []

        for i in range(self._size):
            now_position = fp.tell()
            line = fp.readline()
            try:
                x, y = map(float, line.split())
                self._x.append(x)
                self._y.append(y)
                self._data.append([x, y])
            except:
                # 拟合数据中的数据长度不等于self,_size
                fp.seek(now_position)  # 回到上一行
                self._size = i   # 更新点的数量
                break

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def data(self):
        return self._data

    def plot(self, show=True):
        import matplotlib.pyplot as plt
        df = pd.DataFrame(self._data)
        x = df.iloc[:, 0]
        y = df.iloc[:, 1]
        plt.plot(x, y)
        if show:
            plt.show()

    def get_narray(self):
        import numpy as np
        return np.array([self.x, self.y])


def main():
    fp = open('../../data/origin/before/LINE_100_dbdt.dat', 'r')
    fp.readline()
    fp.readline()
    point_data = SinglePoint(fp)
    point_data.plot()


if __name__ == '__main__':
    main()
