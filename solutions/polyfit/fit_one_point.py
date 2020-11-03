from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from common.unit import SingleFile
from common.unit import SinglePoint

before_filepath = "../../data/origin/before/LINE_100_dbdt.dat"
after_filepath = "../../data/origin/after/new_LINE_100_dbdt.dat"
batch_size = 10

def target_func(input, a, b, c, d, e, f, g, h, i):
    ep = 1e-7
    input = abs(input)
    return a * np.log(abs(b * input + ep)) + c * np.log(abs(d * input + ep)) ** 2 + e * np.log(abs(f * input + ep)) ** 3 + g * np.log(abs(h * input + ep)) ** 4 + i


def fit_one_point(point:SinglePoint, show=False):
    scaler = MinMaxScaler()
    x_data = np.array(point.x)
    x_data = x_data * 100
    y_data = np.array(point.y)
    y_data = scaler.fit_transform(np.array(y_data).reshape(-1, 1) * 10) + 0.2
    y_data = y_data.flatten()

    plt.plot(x_data, y_data, 'b--')

    popt, pcov = curve_fit(target_func, x_data, y_data)
    # print(popt)

    y2 = [target_func(i, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8]) for i in x_data]

    plt.plot(x_data, y2, 'r-')

    if show:
        plt.show()

    return popt

if __name__ == '__main__':
    before_file = SingleFile(before_filepath)
    after_file = SingleFile(after_filepath)

    fit_one_point(before_file.get_one_point(), show=True)
