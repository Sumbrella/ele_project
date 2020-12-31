import numpy as np
import matplotlib.pyplot as plt

from ele_common.functions import loop_tem1d


if __name__ == '__main__':
    # === ORIGIN ===
    L_square = 1
    depth = [100, 200]
    res = [70, 50, 600]

    time_channels = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105]
    plt.figure(figsize=(6, 6))
    for index, channel in enumerate(time_channels):
        times = np.linspace(start=1e-4, stop=1e-2, num=channel)

        EM_b, EM_db = loop_tem1d(times, L_square, depth, res, verb_flag=0)
        plt.subplot(4, 4, index + 1)
        plt.title(f"channels: {channel}", fontsize=8)
        plt.xticks(None)
        plt.plot(times, np.abs(EM_db), 'r--')

    plt.show()
