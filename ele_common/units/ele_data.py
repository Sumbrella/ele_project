import os
import csv

from loguru import logger

from ..functions import is_dir_exist
from ..functions import make_dir_with_input


class EleData:
    def __init__(self, data_dict):

        self.params = [
            'origin_data',
            'result_data',
            'time',
            'layer_number',
            'depths',
            'res',
        ]

        self.origin_data = data_dict['origin_data']
        self.result_data = data_dict['result_data']
        self.time = data_dict['time']
        self.layer_number = data_dict['layer_number']
        self.depths = data_dict['depths']
        self.res = data_dict['res']

    @staticmethod
    def _dir_init(save_dir):
        # 用于保存教师数据
        teacher_dir = os.path.join(save_dir, 'data')
        # 用于保存扰动数据
        data_dir = os.path.join(save_dir, 'teacher')
        # 用于保存其他标签
        label_dir = os.path.join(save_dir, 'label')

        if not is_dir_exist(save_dir):
            make_dir_with_input(path=save_dir)

        if not is_dir_exist(teacher_dir):
            os.mkdir(teacher_dir)

        if not is_dir_exist(data_dir):
            os.mkdir(data_dir)

        if not is_dir_exist(label_dir):
            os.mkdir(label_dir)

    def _add_point_to_file(self, full_path, which):

        if which not in ['origin', 'result']:
            logger.error('param \'which\' should be \'origin \' or \' result \'')
            raise ValueError('param \'which\' should be \'origin \' or \' result \'')

        save_data = None

        if which == 'origin':
            save_data = self.origin_data
        elif which == 'result':
            save_data = self.result_data

        if not os.path.exists(full_path):
            import time
            # 如果不存在文件，则创建文件并写入 0 和 日期
            fp = open(os.path.join(full_path), 'w')
            fp.write(' 0\n')
            fp.write(" ")
            fp.write(time.asctime())
            fp.write("\n")

        with open(os.path.join(full_path), 'r+') as fp:
            # 读取第一行的point数目，并令其+1
            point_number = int(fp.readline())
            fp.seek(0)
            fp.writelines(' {}\n'.format(point_number + 1))

        with open(os.path.join(full_path), 'a+') as fp:
            # 写入一个point
            fp.write(' {}\n'.format(point_number + 1))
            fp.write(' 1   2   3   4\n')
            fp.write(' xy\n')
            fp.write(' {}\n'.format(len(self.time)))

            for x, y in zip(self.time, save_data):
                fp.write(' {:e}\t\t{:e}\n'.format(x, y))

    def _add_point_label(self, full_path):
        with open(full_path, 'a+') as cv:
            fieldnames = ['layer_number', 'depths', 'res']
            writer = csv.DictWriter(cv, fieldnames=fieldnames)
            writer.writerow({
                'layer_number': self.layer_number,
                'depths': self.depths,
                'res': self.res,
            })

    def add_to_dat(self, save_dir, file_name):
        self._dir_init(save_dir)

        teacher_dir = os.path.join(save_dir, 'teacher')
        data_dir = os.path.join(save_dir, 'data')
        label_dir = os.path.join(save_dir, 'label')

        self._add_point_to_file(os.path.join(data_dir, file_name + '.dat'), which='origin')
        self._add_point_to_file(os.path.join(teacher_dir, 'NEW_' + file_name + '.dat'), which='result')
        self._add_point_label(os.path.join(label_dir, file_name + '.csv'))

    def draw(self):
        import matplotlib.pyplot as plt
        plt.plot(
            self.time,
            self.origin_data,
            alpha=1,
            label='origin',
        )
        plt.plot(
            self.time,
            self.result_data,
            alpha=0.7,
            ls='--',
            label="result",
        )
        plt.legend()
        plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from ele_common.units import Generator

    generator = Generator()
    data = generator.generate(debug=True)
    ele_data = EleData(data)

    print(ele_data.layer_number)
    print(ele_data.origin_data)
    print(ele_data.result_data)

    plt.plot(ele_data.time, ele_data.origin_data, label='origin')
    plt.plot(ele_data.time, ele_data.result_data, label='added')
    plt.legend()
    plt.show()
