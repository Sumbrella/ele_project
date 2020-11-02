from typing import TextIO

import pandas as pd
import numpy as np

class SinglePoint:

    def __init__(self, fp:TextIO, skip_line=2):
        id = fp.readline()
        if id is None:
            raise RuntimeError('No more line exist in {}'.format(fp))
        self._id = int(id)

        for i in range(skip_line):
            next(fp)
        self._size = eval(fp.readline())
        self._data = []
        self._x = []
        self._y = []
        for i in range(self._size):
            x, y = map(float, fp.readline().split())
            self._x.append(x)
            self._y.append(y)
            self._data.append([x, y])

    def plot(self):
        import matplotlib.pyplot as plt
        df = pd.DataFrame(self._data)
        x = df.iloc[:, 0]
        y = df.iloc[:, 1]
        plt.plot(x, y)
        plt.show()

    def get_data(self):
        return self._data


def main():
    fp = open('../../data/origin/before/LINE_100_dbdt.dat', 'r')
    fp.readline()
    fp.readline()
    point_data = SinglePoint(fp)
    point_data.plot()


if __name__ == '__main__':
    main()
