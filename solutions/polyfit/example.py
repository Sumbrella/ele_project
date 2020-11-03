from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np


def func(x, a, b, c, d, e):
    return a * np.log(b * (x**2)) + c * np.log(d * x) + e


xdata = np.array([18, 19, 20, 21, 22, 23])
ydata = np.array([196, 239, 294, 444, 590, 850])
plt.plot(xdata, ydata, 'b-')
popt, pcov = curve_fit(func, xdata, ydata)

y2 = [func(i, popt[0], popt[1], popt[2], popt[3], popt[4]) for i in np.linspace(19, 31, 50)]
plt.plot(np.linspace(19, 31, 50), y2, 'r--')
# print(popt)
plt.show()

for date in range(24, 31):
    print("{}日：{:.0f}".format(date, func(date, popt[0], popt[1], popt[2], popt[3], popt[4])))
