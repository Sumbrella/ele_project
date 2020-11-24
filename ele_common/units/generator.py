import sys

import numpy as np
import matplotlib.pyplot as plt

from ele_common.functions import loop_tem1d
from solutions.lstm_with_cnn.config import *


class Generator:
    def __init__(self):
        self.layer_number = None
        self.generate_times = 0

    def generate(self, layer_number=None, debug=False):

        if layer_number is None:
            layer_number = np.random.randint(
                low=default_square_scope[0],
                high=default_square_scope[1],
                size=1
            )

        depth = self._generate_depth(layer_number)

        res = self._generate_res(layer_number)

        square = self._generate_square()

        time = self._generate_time()

        _, db_data = loop_tem1d(time=time, L_square=square, depth=depth, res=res)

        db_data = np.abs(db_data)  # 加绝对值

        # 增加扰动
        res_data = self.add_perturbation(db_data)

        if debug is True:
            print("=====depth=====\n", depth)
            print("=====res=====\n", res)
            print("=====square=====\n", square)
            print("=====time=====\n", time)
            print("=====db_data=====\n", db_data)
            print("=====res_data====\n", res_data)

        return {
            'origin_data': db_data,
            'result_data': res_data,
            'time': time,
            'depths': depth,
            'layer_number': layer_number,
            'res': res,
            'square': square,
        }

    @staticmethod
    def add_perturbation(db_data, perturbation_scope=None):
        if perturbation_scope is None:
            perturbation_scope = default_perturbation_range
        # 增加扰动
        # B 数据 在其原始数据的 (-15 ~ 80) % 之间抖动
        # 扰动因子 k = 历史扰动数据的平方和开方分之一
        res_data = db_data.copy()
        k = 1
        for index, b_data in enumerate(res_data):
            perturbation_rate = np.random.randint(perturbation_scope[0], perturbation_scope[1]) / 100 / (k ** 0.5)
            res_data[index] = b_data * (1 + perturbation_rate)
            k = k + perturbation_rate ** 2
    
        return res_data

    @staticmethod
    def _generate_res(layer_number, scope=None):
        if scope is None:
            scope = default_res_scope

        # 在 0 - 1 内 生成随机数
        pre_res = np.random.random_sample(layer_number + 1)
        # min-max 划入 scope 范围内
        res = scope[0] + pre_res * (scope[1] - scope[0])

        return res

    @staticmethod
    def _generate_depth(layer_number, scope=None):
        if scope is None:
            scope = default_deep_scope

        pre_depth = np.random.random_sample(layer_number)

        # min-max 划入 scope 范围内
        depths = scope[0] + pre_depth * (scope[1] - scope[0])

        # 生成递增序列
        for i, depth in enumerate(depths):
            if i == 0:
                continue
            depths[i] = depth + depths[i - 1]

        return depths

    @staticmethod
    def _generate_square(scope=None):

        if scope is None:
            scope = default_square_scope

        pre_square = np.random.randint(
            default_square_scope[0],
            default_square_scope[1],
        )

        # min-max 划入 scope 范围内
        square = scope[0] + pre_square * (scope[1] - scope[0])

        return square

    @staticmethod
    def _generate_time(time_sequence_number=None):
        if time_sequence_number is None:
            time_sequence_number = np.random.randint(default_time_scope[0], default_time_scope[1])

        times = np.linspace(start=default_time_range[0], stop=default_time_range[1], num=time_sequence_number)

        return times


if __name__ == '__main__':
    generator = Generator()
    data = generator.generate(debug=True)
    plt.plot(data['time'], data['result_data'])
    plt.show()
