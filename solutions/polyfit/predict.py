import os

from paddle import fluid
import numpy as np
import matplotlib.pyplot as plt

from solutions.polyfit.example import AlexNet
from solutions.polyfit.teacher_change import back_change
from ele_common.units import SingleFile
from ele_common.functions.polyfit import exponenial_func


def main():
    data_path = "../../data/train/before/LINE_100_dbdt.dat"

    with fluid.dygraph.guard():
        model = AlexNet()
        # Load static
        min_dict, _ = fluid.load_dygraph(model_path='min_polyfit')
        # print(min_dict)
        model.set_dict(stat_dict=min_dict)

        model.eval()

        data_file = SingleFile(data_path)
        one_point = data_file.get_one_point()

        data = one_point.get_data()

        data = np.array(data, 'float32').reshape(1, 2, 1, 100)
        # data.res
        data = fluid.dygraph.to_variable(data)

        logits = model(data)

        result = logits.numpy()

        result = back_change(result)

    x_data = one_point.x
    print("RESULT: \n", result)
    one_point.plot(show=False, label='origin')
    plt.plot(x_data, [exponenial_func(x, *result[0]) for x in x_data], label='predict')
    plt.show()


if __name__ == '__main__':
    main()
