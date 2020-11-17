from common.functions import loop_tem1d

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    times = np.linspace(0, 1, 1000000)
    L_square = 1
    depth = [20, 70, 100]
    res = [50, 30, 40, 60]
    M_b, EM_db = loop_tem1d(times, L_square, depth, res, verb_flag=0)
    # print(type(M_b))

    plt.plot(times, M_b)
    plt.show()

    plt.plot(times, EM_db)
    plt.show()