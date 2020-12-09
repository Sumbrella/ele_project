from math import log

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
            layer_number = np.random.randint(default_layer_scope[0], default_layer_scope[1])

        depth = self._generate_depth(layer_number)

        res = self._generate_res(layer_number)

        square = self._generate_square()

        time = self._generate_time()

        _, db_data = loop_tem1d(time=time, L_square=square, depth=depth, res=res, verb_flag=0)
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
        res_data = [0 for _ in range(len(db_data))]
        k = 1
        # 方案1： 在原始时间上增加扰动
        # for index, value in enumerate(db_data):
        #     perturbation_rate = np.random.randint(perturbation_scope[0], perturbation_scope[1]) \
        #             / 100 / (k ** 0.5)
        #     res_data[index] = value * (1 + perturbation_rate)
        #     k = k + perturbation_rate ** 2
        #

        # 方案2： 在上一个时间增加扰动
        for index, value in enumerate(db_data):
            last_value = db_data[index-1] if index > 0 else db_data[index]
            perturbation_rate = np.random.randint(
                    perturbation_scope[0], perturbation_scope[1]
            ) / 100

            res_data[index] = last_value * (1 + perturbation_rate)

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

        square = np.random.randint(
            scope[0],
            scope[1],
        )

        return square

    @staticmethod
    def _generate_time(time_sequence_number=None, time_range=None):
        if time_sequence_number is None:
            time_sequence_number = np.random.randint(default_time_scope[0], default_time_scope[1])
        if time_range is None:
            time_range = default_time_range

        # 方案1: 等距取样
        # times = np.linspace(start=default_time_range[0], stop=default_time_range[1], num=time_sequence_number)

        # 方案2: 对数间隔取样
        min_log_time, max_log_time = log(default_time_range[0]), log(default_time_range[1])

        times = np.linspace(start=min_log_time, stop=max_log_time, num=time_sequence_number)
        times = np.exp(times)

        return times


if __name__ == '__main__':
    generator = Generator()

    data = generator.generate(debug=True)

    plt.plot(data['time'], data['origin_data'], label='origin')
    plt.plot(data['time'], data['result_data'], label='result', alpha=0.6)

    plt.legend()
    plt.show()
