import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.preprocessing import MinMaxScaler


def exponenial_func(input, a, b, c, d, e, f, g, h, i):
    ep = 1e-7
    input = abs(input)
    return a * np.log(abs(b * input + ep)) + c * np.log(abs(d * input + ep)) ** 2 + e * np.log(abs(f * input + ep)) ** 3 + g * np.log(abs(h * input + ep)) ** 4 + i


def fit_point(point, show=False, maxfev=1000000):
    scaler = MinMaxScaler()
    x_data = np.array(point.x)
    x_data = x_data * 100
    y_data = np.array(point.y)
    y_data = scaler.fit_transform(np.array(y_data).reshape(-1, 1) * 10) + 0.2
    y_data = y_data.flatten()

    popt, pcov = curve_fit(exponenial_func, x_data, y_data, p0=[1 for _ in range(9)], maxfev=maxfev)
    # print(popt)
    y2 = [exponenial_func(i, *popt) for i in x_data]

    if show:
        plt.plot(x_data, y_data, 'b--', label='origin')
        plt.plot(x_data, y2, 'r--', label='fit')
        plt.legend()
        plt.show()

    return popt


if __name__ == '__main__':
    from ele_common.units import SingleFile, SinglePoint

    before_filepath = "../../data/origin/before/LINE_100_dbdt.dat"
    after_filepath = "../../data/origin/after/new_LINE_100_dbdt.dat"

    before_file = SingleFile(before_filepath)
    after_file = SingleFile(after_filepath)

    fit_point(before_file.get_one_point(), show=True)
